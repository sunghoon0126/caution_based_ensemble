"""Full-pool classifier-only SGD sweep. Verify, extract, then evaluate + plot."""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.run_lr_step_expansion import make_adapter, write_csv
from ensemble.evaluation.baseline6.run_selected_sgd import add_problem, summarize
from ensemble.models.scalar_reward_inputs import tokenize_rewards

LRS=(1e-8,1e-7,1e-6,1e-5)
STEPS=(1,2,4,8,16)
NS=tuple(2**i for i in range(10))
N=512
RM="OpenAssistant/reward-model-deberta-v3-large-v2"
ARMS=[(f"step{k}_lr{lr:g}",lr,k) for lr in LRS for k in STEPS]


def config(args):
    return dict(version=1,original=str(args.original.resolve()),dataset=args.dataset,
                problems=args.problems,dummy_pool=N,scored_candidates=N,
                reward_model=RM,reward_backend="auto",dtype="float32",
                revision="main",max_length=512,scope="classifier.weight_and_bias_only",
                optimizer="sgd",momentum=0,weight_decay=0,loss="half_mse",
                lrs=list(LRS),steps=list(STEPS),world_size=args.world_size,
                batch_size=args.batch_size)


def prepare(args):
    if args.original is None or not args.original.is_file():
        raise ValueError("Expected existing --original score file with response text")
    c=config(args)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    p=args.output_dir/"last_layer_config.json"
    if p.exists():
        if json.loads(p.read_text())!=c:
            raise ValueError("Changed manifest; use a new output directory")
    else:
        p.write_text(json.dumps(c,indent=2))
    return c


def check_group(batch,c,p):
    expected={"backend":"pair","dtype":"float32","revision":"main","max_length":512}
    if (int(batch[0]["dataset_index"])!=p or any(
        r["dataset"]!=c["dataset"] or r["split"]!="test"
        or r.get("reward_model_path")!=RM or r.get("reward_scoring_config")!=expected
        or r["correct"] not in (0,1) for r in batch)):
        raise ValueError(f"Unexpected original data at problem {p}")


def load_head(c,args,row):
    import torch
    adapter=make_adapter(c,args.device,row)
    head=getattr(adapter.model,"classifier",None)
    if (adapter.model.config.model_type!="deberta-v2"
        or not isinstance(head,torch.nn.Linear) or head.out_features!=1
        or head.bias is None or head.weight.dtype!=torch.float32):
        raise ValueError("Expected FP32 scalar linear classifier with bias")
    adapter.model.eval()
    adapter.model.requires_grad_(False)
    return adapter,head


def features(adapter,head,rows):
    import torch
    captured=[]
    hook=head.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
    try:
        tokens=tokenize_rewards(adapter.tokenizer,rows,adapter.settings["backend"],
                                adapter.settings["max_length"])
        tokens={k:v.to(adapter.device) for k,v in tokens.items()}
        with torch.no_grad():
            y=adapter.model(**tokens).logits.reshape(-1)
            if len(captured)!=1:
                raise ValueError("Classifier must execute once")
            h=captured[0]
            if h.shape!=(len(rows),head.in_features) or not torch.allclose(
                y,head(h).reshape(-1),rtol=1e-6,atol=1e-6):
                raise ValueError("Unexpected classifier input/output")
            a=h.double().square().sum(1)+1
        return y.cpu().numpy(),a.cpu().numpy(),h
    finally:
        hook.remove()


def matches(rows,pred):
    return len(rows)==len(pred) and all(math.isclose(
        finite(r["reward_model_score"]),finite(v),rel_tol=1e-4,abs_tol=1e-4)
        for r,v in zip(rows,pred))


def verify(args):
    import torch
    c=prepare(args)
    with_group=groups(Path(c["original"]),N)
    try:
        batch=next(with_group)
    finally:
        with_group.close()
    check_group(batch,c,0)
    orig=[finite(r["reward_model_score"]) for r in batch]
    dummy=min(orig)
    adapter,head=load_head(c,args,batch[0])
    ids=sorted(set(np.linspace(0,N-1,min(N,args.verify_candidates),dtype=int).tolist()
                   +[int(np.argmin(orig)),int(np.argmax(orig))]))
    records=[]
    for i in ids:
        y,a,h=features(adapter,head,[batch[i]])
        if not matches([batch[i]],y):
            raise ValueError("Original reward mismatch in verification")
        r=float(y[0]); a=float(a[0])
        for lr in LRS:
            layer=torch.nn.Linear(head.in_features,1,bias=True,device=h.device,dtype=h.dtype)
            layer.load_state_dict(head.state_dict())
            opt=torch.optim.SGD(layer.parameters(),lr=lr,momentum=0,weight_decay=0)
            for k in range(1,17):
                opt.zero_grad(set_to_none=True)
                loss=.5*(layer(h).reshape(())-dummy).square()
                loss.backward()
                opt.step()
                if k in STEPS:
                    actual=finite(layer(h).detach().item())
                    analytic=finite(dummy+(r-dummy)*(1-lr*a)**k)
                    records.append(dict(candidate_id=batch[i]["candidate_id"],learning_rate=lr,
                        steps=k,explicit=actual,analytic=analytic,absolute_error=abs(actual-analytic),
                        feature_norm_sq_plus_bias=a,effective_lr=lr*a))
                    if not math.isclose(actual,analytic,rel_tol=1e-4,abs_tol=2e-4):
                        raise ValueError(f"Explicit SGD mismatch: {records[-1]}")
    write_csv(args.output_dir/"verification.csv",records)
    proof=dict(config=c,model_revision=getattr(adapter.model.config,"_commit_hash",None),
               num_candidates=len(ids),max_absolute_error=max(r["absolute_error"] for r in records),
               max_effective_lr=max(r["effective_lr"] for r in records),
               classifier_parameters=head.weight.numel()+head.bias.numel())
    (args.output_dir/"verification.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2),flush=True)


def shard(out,rank):
    return out/f"features_rank_{rank}.jsonl"


def shard_config(c,rank):
    return dict(experiment=c,rank=rank)


def extract(args):
    c=prepare(args)
    proof=json.loads((args.output_dir/"verification.json").read_text())
    if proof["config"]!=c:
        raise ValueError("Verification config mismatch")
    rank=args.rank
    if not 0<=rank<c["world_size"]:
        raise ValueError("Invalid rank")
    writer=ResumeWriter(shard(args.output_dir,rank),shard_config(c,rank))
    adapter=head=None
    done=new=seen=0
    total=len(range(rank,c["problems"],c["world_size"]))*N
    started=time.monotonic()
    next_log=args.log_every
    try:
        for p,batch in enumerate(groups(Path(c["original"]),N)):
            if p>=c["problems"]:
                raise ValueError("Source has more problems than requested")
            check_group(batch,c,p)
            seen+=1
            if p%c["world_size"]!=rank:
                continue
            dummy=min(finite(r["reward_model_score"]) for r in batch)
            context=dict(dummy_reward=dummy,pool_hash=digest(batch))
            for j in range(0,N,c["batch_size"]):
                pending=[]
                for row in batch[j:j+c["batch_size"]]:
                    completed,prov=writer.completed(row,context)
                    if completed:
                        done+=1
                    else:
                        pending.append((row,prov))
                if pending:
                    if adapter is None:
                        adapter,head=load_head(c,args,pending[0][0])
                        if getattr(adapter.model.config,"_commit_hash",None)!=proof["model_revision"]:
                            raise ValueError("Model revision differs from verification")
                    rows=[r for r,_ in pending]
                    pred,norms,h=features(adapter,head,rows)
                    del h
                    if not matches(rows,pred):
                        # Fall back to original single-candidate padding behavior.
                        pairs=[]
                        for row in rows:
                            pr,no,h=features(adapter,head,[row])
                            del h
                            if not matches([row],pr):
                                raise ValueError(f"Original score mismatch: {row['candidate_id']}")
                            pairs.append((pr[0],no[0]))
                        pred,norms=zip(*pairs)
                    for (row,prov),pr,a in zip(pending,pred,norms):
                        if row.get("reward_model_revision") and row["reward_model_revision"]!=proof["model_revision"]:
                            raise ValueError("Mixed checkpoint revisions")
                        compact={k:row[k] for k in ("dataset","split","trial_id","dataset_index",
                                 "instance_id","sample_index","candidate_id","correct","reward_model_score")}
                        writer.write(dict(compact,**prov,finite=True,dummy_reward=dummy,
                            reward_after=finite(row["reward_model_score"]),forward_reward=finite(pr),
                            feature_norm_sq_plus_bias=finite(a)))
                        done+=1; new+=1
                if done>=next_log or done==total:
                    rate=new/max(time.monotonic()-started,1e-9)
                    eta=(total-done)/rate/60 if rate else float("inf")
                    print(f"[rank {rank}] {done:,}/{total:,} ({100*done/total:.2f}%) "
                          f"new={new:,} new_avg={rate:.3f} cand/s ETA={eta:.1f} min",flush=True)
                    next_log=(done//args.log_every+1)*args.log_every
        writer.finish()
        if seen!=c["problems"] or done!=total:
            raise ValueError(f"Incomplete extraction: {seen} problems, {done} candidates")
    finally:
        writer.close()
    print(f"[rank {rank}] Complete",flush=True)


def auc(scores,truth):
    pos=~truth
    npos=int(pos.sum()); nneg=len(truth)-npos
    if not npos or not nneg:
        return float("nan")
    _,inv,cnt=np.unique(scores,return_inverse=True,return_counts=True)
    ranks=np.cumsum(cnt)-(cnt-1)/2
    return float((ranks[inv[pos]].sum()-npos*(npos+1)/2)/(npos*nneg))


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def evaluate(args):
    c=json.loads((args.output_dir/"last_layer_config.json").read_text())
    proof=json.loads((args.output_dir/"verification.json").read_text())
    if c!=proof["config"]:
        raise ValueError("Verification config mismatch")
    lrs=sorted(set(args.eval_lrs if args.eval_lrs is not None else c["lrs"]))
    if not lrs or any(not math.isfinite(lr) or lr<=0 for lr in lrs):
        raise ValueError("Evaluation learning rates must be positive and finite")
    out=(args.evaluation_dir or args.output_dir/"evaluation").resolve()
    if lrs!=sorted(c["lrs"]) and out==(args.output_dir/"evaluation").resolve():
        raise ValueError("Expanded LR sweep requires --evaluation-dir to preserve old results")
    if out==args.output_dir.resolve():
        raise ValueError("Evaluation directory must differ from feature directory")
    out.mkdir(parents=True,exist_ok=True)
    eval_config=dict(feature_config=c,lrs=lrs,steps=c["steps"],
                     calculation="closed_form_float64",source_dir=str(args.output_dir.resolve()))
    meta=out/"evaluation_config.json"
    if meta.exists() and json.loads(meta.read_text())!=eval_config:
        raise ValueError("Evaluation config changed; choose a different evaluation directory")
    meta.write_text(json.dumps(eval_config,indent=2))
    arms=[(f"step{k}_lr{lr:g}",lr,k) for lr in lrs for k in c["steps"]]
    methods=["vanilla"]+[m for m,_,_ in arms]
    counts,totals=defaultdict(Counter),Counter()
    size=c["problems"]*N
    pen=np.empty((len(arms),size),dtype=np.float64)
    rel=np.empty_like(pen)
    truth=np.empty(size,dtype=bool)
    norms=np.empty(size,dtype=np.float64)
    top=np.empty(c["problems"]*16,dtype=np.int64)
    sources=[iter(groups(shard(args.output_dir,r),N)) for r in range(c["world_size"])]
    try:
        for p in range(c["problems"]):
            rank=p%c["world_size"]
            batch=next(sources[rank],None)
            if batch is None or int(batch[0]["dataset_index"])!=p:
                raise ValueError(f"Missing feature group {p}")
            if any(r["backprop_config"]!=shard_config(c,rank) or not r.get("finite")
                   or r["dataset"]!=c["dataset"] or r["correct"] not in (0,1) for r in batch):
                raise ValueError("Invalid feature cache")
            orig=np.array([finite(r["reward_model_score"]) for r in batch])
            dummy=float(orig.min())
            if any(r["dummy_reward"]!=dummy for r in batch):
                raise ValueError("Dummy reward mismatch")
            a=np.array([finite(r["feature_norm_sq_plus_bias"]) for r in batch])
            if np.any(a<1):
                raise ValueError("Invalid feature norm")
            t=np.array([bool(r["correct"]) for r in batch])
            ids=slice(p*N,(p+1)*N)
            truth[ids]=t; norms[ids]=a
            top[p*16:(p+1)*16]=p*N+np.argsort(-orig,kind="stable")[:16]
            scores={"vanilla":orig}
            gap=orig-dummy
            for j,(m,lr,k) in enumerate(arms):
                factor=1-(1-lr*a)**k
                v=gap*factor
                if not np.isfinite(v).all():
                    raise ValueError(f"Nonfinite penalty for {m}")
                pen[j,ids]=v
                rel[j,ids]=np.where(gap>0,factor,np.nan)
                scores[m]=orig-v
            add_problem(counts,totals,("all",),t,scores,NS)
            if (p+1)%50==0:
                print(f"Evaluated {p+1}/{c['problems']} problems",flush=True)
        if any(next(s,None) is not None for s in sources):
            raise ValueError("Extra feature groups")
    finally:
        for s in sources:
            s.close()
    curves=summarize(counts,totals,methods,NS)
    lookup={m:(lr,k) for m,lr,k in arms}
    for r in curves:
        r["learning_rate"],r["steps"]=lookup.get(r["method"],(0.,0))
    summary=[]
    for m in methods:
        pts=[r for r in curves if r["method"]==m]
        peak=max(pts,key=lambda r:r["accuracy_percent"]); end=pts[-1]
        summary.append(dict(subset="all",method=m,learning_rate=end["learning_rate"],
            steps=end["steps"],peak_n=peak["N"],peak_accuracy_percent=peak["accuracy_percent"],
            final_n=N,final_accuracy_percent=end["accuracy_percent"],
            final_delta_pp=end["delta_vs_vanilla_pp"],
            degradation_pp=peak["accuracy_percent"]-end["accuracy_percent"],
            final_flip_rate_percent=end["flip_rate_percent"],
            final_I_to_C=end["I_to_C"],final_C_to_I=end["C_to_I"]))
    write_csv(out/"curves.csv",curves)
    write_csv(out/"summary.csv",summary)
    stats=[]
    for scope,ids in (("all",np.arange(size)),("top16_original",top)):
        t=truth[ids]
        for j,(m,lr,k) in enumerate(arms):
            v=pen[j,ids]; f=rel[j,ids]
            good,bad=v[t],v[~t]
            g,b=f[t],f[~t]
            gm,bm=mean(g[np.isfinite(g)]),mean(b[np.isfinite(b)])
            stats.append(dict(scope=scope,method=m,learning_rate=lr,steps=k,
                n_correct=len(good),n_incorrect=len(bad),mean_correct=mean(good),
                mean_incorrect=mean(bad),delta_incorrect_minus_correct=mean(bad)-mean(good),
                std_correct=float(good.std()) if len(good) else float("nan"),
                std_incorrect=float(bad.std()) if len(bad) else float("nan"),
                relative_mean_correct=gm,relative_mean_incorrect=bm,relative_delta=bm-gm,
                penalty_auroc=auc(v,t),negative_fraction_correct=mean(good<0),
                negative_fraction_incorrect=mean(bad<0),
                unstable_fraction=mean(lr*norms[ids]>=2),
                overshoot_fraction=mean(lr*norms[ids]>1)))
    write_csv(out/"penalty_by_lr_step.csv",stats)
    stability=[dict(learning_rate=lr,effective_lr_min=float(lr*norms.min()),
        effective_lr_median=float(lr*np.median(norms)),
        effective_lr_p95=float(lr*np.quantile(norms,.95)),
        effective_lr_p99=float(lr*np.quantile(norms,.99)),
        effective_lr_max=float(lr*norms.max()),
        overshoot_fraction=mean(lr*norms>1),
        unstable_fraction=mean(lr*norms>=2)) for lr in lrs]
    write_csv(out/"stability.csv",stability)
    for row in stability:
        print(f"LR={row['learning_rate']:g} effective max={row['effective_lr_max']:.6g} "
              f"overshoot={100*row['overshoot_fraction']:.3f}% "
              f"unstable={100*row['unstable_fraction']:.3f}%",flush=True)
    from ensemble.evaluation.baseline6.plot_last_layer_sweep import plot_all
    plot_all(out,curves,summary,stats,dict(c,lrs=lrs))
    stable_lrs=[r["learning_rate"] for r in stability if r["unstable_fraction"]==0]
    if stable_lrs and len(stable_lrs)!=len(lrs):
        stable=out/"stable_only"; stable.mkdir(exist_ok=True)
        keep=lambda r:r["method"]=="vanilla" or r["learning_rate"] in stable_lrs
        cs=[r for r in curves if keep(r)]
        ss=[r for r in summary if keep(r)]
        ps=[r for r in stats if keep(r)]
        write_csv(stable/"curves.csv",cs)
        write_csv(stable/"summary.csv",ss)
        write_csv(stable/"penalty_by_lr_step.csv",ps)
        plot_all(stable,cs,ss,ps,dict(c,lrs=stable_lrs))
    if not stable_lrs:
        print("WARNING: No LR stable for every candidate.",flush=True)
    for r in sorted(summary,key=lambda r:-r["final_accuracy_percent"])[:6]:
        print(r,flush=True)
    print(f"Saved all results and plots to {out}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("verify","extract","evaluate"))
    p.add_argument("--original",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--dataset",choices=("gsm8k","logiqa2","arc_challenge"),default="gsm8k")
    p.add_argument("--problems",type=int,default=1000)
    p.add_argument("--world-size",type=int,choices=(1,2,3,4),default=1)
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--verify-candidates",type=int,default=8)
    p.add_argument("--log-every",type=int,default=100)
    p.add_argument("--eval-lrs",type=float,nargs="+",help="CPU evaluation LR override; cached feature manifest is unchanged")
    p.add_argument("--evaluation-dir",type=Path,help="Separate directory for evaluation outputs")
    args=p.parse_args()
    if min(args.problems,args.batch_size,args.verify_candidates,args.log_every)<1:
        p.error("Positive sizes required")
    if args.mode!="evaluate" and (args.eval_lrs is not None or args.evaluation_dir is not None):
        p.error("--eval-lrs and --evaluation-dir are evaluation-only options")
    args.output_dir=args.output_dir.resolve()
    {"verify":verify,"extract":extract,"evaluate":evaluate}[args.mode](args)


if __name__=="__main__":
    main()
