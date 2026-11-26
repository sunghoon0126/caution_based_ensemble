"""Full-model FP32 SGD, one independent LR per GPU; resume and evaluate."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ensemble.evaluation.baseline6.common import ResumeWriter, digest, finite, groups
from ensemble.evaluation.baseline6.run_lr_step_expansion import make_adapter, write_csv
from ensemble.evaluation.baseline6.run_selected_sgd import add_problem, summarize
from ensemble.evaluation.baseline6.run_last_layer_sweep import auc, mean

LRS=(1e-8,1e-6,1e-4,3e-3)
STEPS=(1,2,4,8,16)
N=512
NS=[2**i for i in range(10)]
RM="OpenAssistant/reward-model-deberta-v3-large-v2"


def config(args):
    return dict(version=1,original=str(args.original.resolve()),dataset="gsm8k",
                problems=args.problems,dummy_pool=N,scored_candidates=N,
                reward_model=RM,reward_backend="auto",dtype="float32",
                revision="main",max_length=512,scope="full",
                scope_label="full-model SGD",optimizer="sgd",momentum=0,
                weight_decay=0,loss="half_mse",lrs=list(LRS),steps=list(STEPS))


def manifest(args):
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    p=out/"full_model_config.json"
    if args.mode=="launch":
        if args.original is None or not args.original.is_file():
            raise ValueError("Missing original RM scores")
        c=config(args)
        if p.exists() and json.loads(p.read_text())!=c:
            raise ValueError("Manifest mismatch; use a fresh output directory")
        if not p.exists():
            p.write_text(json.dumps(c,indent=2))
    return json.loads(p.read_text())


def shard(out,rank):
    return out/f"rank_{rank}.jsonl"


def shard_config(c,rank):
    return dict(experiment=c,rank=rank,learning_rate=LRS[rank])


def adapt(model,inputs,target,lr):
    """Same full-model stateless SGD as Test 1; retain finite prefix on divergence."""
    import torch
    from torch.func import functional_call
    model.eval()
    params={k:v.detach().requires_grad_(True) for k,v in model.named_parameters()}
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    def forward():
        result=functional_call(model,(params,buffers),(),inputs,strict=True)
        if result.logits.numel()!=1:
            raise ValueError("Expected scalar reward")
        return result.logits.reshape(()).float()
    scores={}
    with torch.enable_grad():
        reward=forward()
        before=finite(reward.detach().item())
        for k in range(1,17):
            try:
                loss=.5*(reward-target).square()
                finite(loss.detach().item())
                grads=torch.autograd.grad(loss,list(params.values()),allow_unused=True)
                updated={}
                with torch.no_grad():
                    for (name,v),g in zip(params.items(),grads):
                        value=v if g is None else v-lr*g
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(f"Nonfinite parameter {name}")
                        updated[name]=value.detach().requires_grad_(True)
                params=updated
                del loss,grads,updated
                reward=forward()
                value=finite(reward.detach().item())
                if k in STEPS:
                    scores[str(k)]=value
            except FloatingPointError as exc:
                return before,scores,dict(step=k,error=str(exc))
    return before,scores,None


def progress(out,rank,done,total,new,started,failures,status):
    elapsed=max(time.monotonic()-started,1e-9)
    rate=new/elapsed
    obj=dict(rank=rank,done=done,total=total,new=new,rate=rate,
             eta=(total-done)/rate if rate else None,failures=failures,status=status)
    p=out/f"progress_{rank}.json"
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


def worker(args):
    c=manifest(args)
    rank=args.rank
    lr=LRS[rank]
    out=args.output_dir
    w=ResumeWriter(shard(out,rank),shard_config(c,rank))
    adapter=None
    done=new=failures=0
    total=c["problems"]*N
    start=time.monotonic()
    status="running"
    seen=0
    progress(out,rank,0,total,0,start,0,"resuming/loading")
    try:
        for p,batch in enumerate(groups(Path(c["original"]),N)):
            if p>=c["problems"]:
                break
            if int(batch[0]["dataset_index"])!=p or any(
                r["dataset"]!="gsm8k" or r["split"]!="test"
                or r.get("reward_model_path")!=RM
                or r.get("reward_scoring_config")!={"backend":"pair","dtype":"float32",
                                                       "revision":"main","max_length":512}
                or r["correct"] not in (0,1) for r in batch):
                raise ValueError(f"Invalid source group {p}")
            seen+=1
            dummy=min(finite(r["reward_model_score"]) for r in batch)
            ctx=dict(dummy_reward=dummy,pool_hash=digest(batch))
            for row in batch:
                completed,prov=w.completed(row,ctx)
                if not completed:
                    if adapter is None:
                        adapter=make_adapter(c,args.device,row)
                        print(f"LR={lr:g} RM loaded",flush=True)
                    rev=row.get("reward_model_revision")
                    if rev and rev!=getattr(adapter.model.config,"_commit_hash",None):
                        raise ValueError("RM checkpoint revision mismatch")
                    before,scores,failure=adapt(adapter.model,adapter.inputs(row["prompt"],row["response"]),dummy,lr)
                    orig=finite(row["reward_model_score"])
                    if not math.isclose(before,orig,rel_tol=1e-4,abs_tol=1e-4):
                        raise ValueError(f"Original reward mismatch: {row['candidate_id']}")
                    compact={k:row[k] for k in ("dataset","split","trial_id","dataset_index",
                            "instance_id","sample_index","candidate_id","correct","reward_model_score")}
                    # finite and reward_after are ResumeWriter record-integrity fields;
                    # actual checkpoint validity is exclusively in scores_by_step.
                    w.write(dict(compact,**prov,finite=True,dummy_reward=dummy,
                                 reward_after=next(reversed(scores.values())) if scores else orig,
                                 scores_by_step=scores,failure=failure))
                    new+=1
                    failures+=int(failure is not None)
                done+=1
                if done<=3 or done%args.log_every==0 or done==total:
                    progress(out,rank,done,total,new,start,failures,"running")
                    print(f"{done}/{total} new={new} new_failures={failures}",flush=True)
        w.finish()
        if seen!=c["problems"] or done!=total:
            raise ValueError("Incomplete source")
        status="complete"
    except BaseException:
        status="interrupted/error"
        raise
    finally:
        w.close()
        progress(out,rank,done,total,new,start,failures,status)


def evaluate(args):
    c=manifest(args); out=args.output_dir/"evaluation"; out.mkdir(exist_ok=True)
    arms=[(f"step{k}_lr{lr:g}",lr,k) for lr in LRS for k in STEPS]
    size=c["problems"]*N
    orig=np.empty(size); truth=np.empty(size,dtype=bool)
    updated=np.full((len(arms),size),np.nan)
    top=np.empty(c["problems"]*16,dtype=int)
    sources=[iter(groups(shard(args.output_dir,r),N)) for r in range(4)]
    try:
        for p in range(c["problems"]):
            ids=slice(p*N,(p+1)*N)
            first=None
            for rank,src in enumerate(sources):
                batch=next(src,None)
                if batch is None or int(batch[0]["dataset_index"])!=p:
                    raise ValueError(f"Missing rank {rank} problem {p}")
                if any(r["backprop_config"]!=shard_config(c,rank) or not r["finite"] for r in batch):
                    raise ValueError("Invalid cache")
                o=np.array([finite(r["reward_model_score"]) for r in batch])
                t=np.array([bool(r["correct"]) for r in batch])
                keys=[r["input_fingerprint"] for r in batch]
                if first is None:
                    first=keys; orig[ids]=o; truth[ids]=t
                    top[p*16:(p+1)*16]=p*N+np.argsort(-o,kind="stable")[:16]
                elif first!=keys or not np.array_equal(o,orig[ids]) or not np.array_equal(t,truth[ids]):
                    raise ValueError("Rank alignment mismatch")
                for j,k in enumerate(STEPS):
                    updated[rank*len(STEPS)+j,ids]=[
                        finite(r["scores_by_step"][str(k)]) if str(k) in r["scores_by_step"] else np.nan
                        for r in batch]
            if (p+1)%50==0:
                print(f"Loaded {p+1}/{c['problems']} problems",flush=True)
        if any(next(src,None) is not None for src in sources):
            raise ValueError("Extra cached groups")
    finally:
        for src in sources:
            src.close()
    valid=np.isfinite(updated).all(axis=1)
    coverage=[dict(method=m,learning_rate=lr,steps=k,valid_candidates=int(np.isfinite(updated[j]).sum()),
                   invalid_candidates=int((~np.isfinite(updated[j])).sum()),
                   total_candidates=size,eligible=bool(valid[j]))
              for j,(m,lr,k) in enumerate(arms)]
    write_csv(out/"checkpoint_coverage.csv",coverage)
    methods=["vanilla"]+[a[0] for j,a in enumerate(arms) if valid[j]]
    counts,totals=defaultdict(Counter),Counter()
    for p in range(c["problems"]):
        ids=slice(p*N,(p+1)*N)
        scores={"vanilla":orig[ids]}
        scores.update({m:updated[j,ids] for j,(m,_,_) in enumerate(arms) if valid[j]})
        add_problem(counts,totals,("all",),truth[ids],scores,NS)
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
            final_n=N,final_accuracy_percent=end["accuracy_percent"],final_delta_pp=end["delta_vs_vanilla_pp"],
            degradation_pp=peak["accuracy_percent"]-end["accuracy_percent"],
            final_flip_rate_percent=end["flip_rate_percent"],
            final_I_to_C=end["I_to_C"],final_C_to_I=end["C_to_I"]))
    write_csv(out/"curves.csv",curves); write_csv(out/"summary.csv",summary)
    gap=(orig.reshape(-1,N)-orig.reshape(-1,N).min(axis=1,keepdims=True)).ravel()
    stats=[]
    for scope,ids in (("all",np.arange(size)),("top16_original",top)):
        t=truth[ids]
        for j,(m,lr,k) in enumerate(arms):
            if not valid[j]:
                continue
            v=orig[ids]-updated[j,ids]
            g=gap[ids]
            rel=np.divide(v,g,out=np.full_like(v,np.nan),where=g>0)
            good,bad=v[t],v[~t]
            rc,rw=rel[t],rel[~t]
            cm,wm=mean(good),mean(bad)
            rcm,rwm=mean(rc[np.isfinite(rc)]),mean(rw[np.isfinite(rw)])
            stats.append(dict(scope=scope,method=m,learning_rate=lr,steps=k,
                n_correct=len(good),n_incorrect=len(bad),mean_correct=cm,mean_incorrect=wm,
                delta_incorrect_minus_correct=wm-cm,std_correct=float(good.std()),
                std_incorrect=float(bad.std()),relative_mean_correct=rcm,relative_mean_incorrect=rwm,
                relative_delta=rwm-rcm,penalty_auroc=auc(v,t)))
    if stats:
        write_csv(out/"penalty_by_lr_step.csv",stats)
        from ensemble.evaluation.baseline6.plot_last_layer_sweep import plot_all
        plot_all(out,curves,summary,stats,c)
    else:
        print("No fully finite configuration; no sweep plots generated",flush=True)
    print(f"Valid configurations: {int(valid.sum())}/{len(arms)}; results: {out}",flush=True)


def launch(args):
    c=manifest(args); out=args.output_dir
    gpu_ids=args.gpus.split(",")
    if len(gpu_ids)!=4 or len(set(gpu_ids))!=4 or any(not x.strip().isdigit() for x in gpu_ids):
        raise ValueError("--gpus requires four distinct GPU indices")
    processes=[]; logs=[]
    module="ensemble.evaluation.baseline6.run_full_model_lr_sweep"
    tty=sys.stdout.isatty()
    try:
        for rank,gpu in enumerate(gpu_ids):
            # Prevent stale status from a prior run from appearing as current.
            (out/f"progress_{rank}.json").unlink(missing_ok=True)
            log=(out/f"gpu{gpu}.log").open("a",buffering=1)
            logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1")
            processes.append(subprocess.Popen([sys.executable,"-u","-m",module,"worker",
                "--output-dir",str(out),"--rank",str(rank),"--device","cuda:0",
                "--log-every",str(args.log_every)],env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True))
        while True:
            lines=[]; total_done=0
            for rank,(gpu,proc) in enumerate(zip(gpu_ids,processes)):
                try:
                    q=json.loads((out/f"progress_{rank}.json").read_text())
                except (FileNotFoundError,json.JSONDecodeError):
                    q={}
                total=c["problems"]*N; done=q.get("done",0)
                total_done+=done; pct=done/total
                bar="#"*int(24*pct)+"-"*(24-int(24*pct))
                eta=q.get("eta"); eta_text=f"{eta/3600:.2f}h" if eta is not None else "--"
                status=("exit="+str(proc.returncode)) if proc.poll() is not None else q.get("status","starting")
                lines.append(f"GPU {gpu} LR={LRS[rank]:g} [{bar}] {pct:6.2%} {done:,}/{total:,} "
                             f"new={q.get('new',0):,} {q.get('rate',0):.3f}/s ETA={eta_text} "
                             f"new failures={q.get('failures',0)} {status}")
            lines.append(f"TOTAL: {total_done:,}/{4*c['problems']*N:,} LR-candidate trajectories")
            if tty:
                sys.stdout.write("\x1b[2J\x1b[H")
            print("\n".join(lines),flush=True)
            if all(p.poll() is not None for p in processes):
                break
            time.sleep(2 if tty else 30)
        if any(p.returncode!=0 for p in processes):
            raise RuntimeError("One or more GPU workers failed; inspect gpu logs; relaunch to resume")
    except BaseException:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        for proc in processes:
            proc.wait()
        raise
    finally:
        for log in logs:
            log.close()
    evaluate(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("launch","worker","evaluate"))
    p.add_argument("--original",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--problems",type=int,default=1000)
    p.add_argument("--gpus",default="0,1,2,3")
    p.add_argument("--rank",type=int,choices=range(4),default=0)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--log-every",type=int,default=10)
    args=p.parse_args()
    if min(args.problems,args.log_every)<1:
        p.error("Positive sizes required")
    args.output_dir=args.output_dir.resolve()
    {"launch":launch,"worker":worker,"evaluate":evaluate}[args.mode](args)


if __name__=="__main__":
    main()
