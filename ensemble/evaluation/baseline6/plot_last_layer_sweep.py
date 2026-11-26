"""Automatic plots for the full classifier-only sweep."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable


def save(fig,out,name):
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/f"{name}.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def plot_all(out,curves,summary,stats,cfg):
    lrs=sorted(cfg["lrs"])
    steps=sorted(cfg["steps"])
    tag=f"{cfg['dataset']} | All {cfg['problems']} problems | {cfg.get('scope_label', 'classifier only')}"
    arms=sorted([r for r in summary if r["method"]!="vanilla"],
                key=lambda r:(-r["final_accuracy_percent"],r["steps"],r["learning_rate"]))
    for name,chosen in (("all",arms),("top5",arms[:5])):
        fig,ax=plt.subplots(figsize=(14, max(7, 5.5+len(chosen)*.12)))
        methods=["vanilla"]+[r["method"] for r in chosen]
        handles=[]
        colors=list(plt.get_cmap("tab20").colors)
        for i,m in enumerate(methods):
            pts=sorted([r for r in curves if r["method"]==m],key=lambda r:r["N"])
            x=[r["N"] for r in pts]; y=[r["accuracy_percent"] for r in pts]
            color="#59677b" if i==0 else colors[(i-1)%len(colors)]
            style=("-","--","-.",":")[((i-1)//len(colors)+i)%4]
            marker=("o","s","^","D","X","v")[i%6]
            if i==0:
                ax.plot(x,y,color=color,linewidth=6,alpha=.12)
            ax.plot(x,y,color=color,linestyle=style,linewidth=1.65,alpha=.95)
            offset=(i-(len(methods)-1)/2)*3.5 if name=="top5" else 0
            ax.plot(x,y,linestyle="none",marker=marker,color=color,markersize=6 if name=="top5" else 3,
                    markeredgecolor="white",markeredgewidth=.65,
                    transform=ax.transData+ScaledTranslation(offset/72,0,fig.dpi_scale_trans),zorder=10+i)
            handles.append(Line2D([0],[0],color=color,linestyle=style,marker=marker,
                    label=f"{m} | {y[-1]:.2f}%"))
        ax.set_xscale("log",base=2)
        ax.set_xticks(x,[str(n) for n in x])
        ax.set_xlim(x[0]/1.25,x[-1]*1.25)
        ax.set(xlabel="Best-of-N",ylabel="Accuracy (%)",title=f"Accuracy vs N | {tag}")
        ax.grid(alpha=.2)
        ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,-.13),
                  ncol=4 if name=="all" else 2,fontsize=8)
        save(fig,out,f"accuracy_vs_n_{name}")

    def matrix(rows,key):
        by={(r["learning_rate"],r["steps"]):r[key] for r in rows}
        return np.array([[by.get((lr,k),np.nan) for lr in lrs] for k in steps],dtype=float)

    def setup(ax,title):
        ax.set_xticks(range(len(lrs)),[f"{v:g}" for v in lrs])
        ax.set_yticks(range(len(steps)),[str(k) for k in steps])
        ax.set(xlabel="Learning rate",ylabel="SGD steps",title=title+"\n"+tag)
        ax.set_xlim(-.5,len(lrs)-.5); ax.set_ylim(-.5,len(steps)-.5)
        ax.set_xticks(np.arange(-.5,len(lrs),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(steps),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=1.3)
        ax.tick_params(which="minor",length=0)

    def norm_for(vals,diverge=False):
        v=np.asarray(vals); v=v[np.isfinite(v)]
        if not len(v):
            return Normalize(0,1)
        if diverge:
            e=max(float(np.max(np.abs(v))),1e-12)
            return TwoSlopeNorm(vmin=-e,vcenter=0,vmax=e)
        lo,hi=float(v.min()),float(v.max())
        if lo==hi:
            lo-=max(abs(lo)*.01,1e-12); hi+=max(abs(hi)*.01,1e-12)
        return Normalize(lo,hi)

    def heat(rows,key,name,title,unit,diverge=False,norm=None,best=False,fmt=".4g"):
        vals=matrix(rows,key)
        norm=norm if norm is not None else norm_for(vals,diverge)
        cmap=plt.get_cmap("RdBu" if diverge else "Blues").copy()
        cmap.set_bad("#eeeeee")
        fig,ax=plt.subplots(figsize=(max(8,1.25*len(lrs)+2),5.8))
        im=ax.imshow(np.ma.masked_invalid(vals),origin="lower",aspect="auto",cmap=cmap,norm=norm)
        for i,j in np.ndindex(vals.shape):
            v=vals[i,j]
            if not np.isfinite(v):
                ax.text(j,i,"—",ha="center",va="center")
                continue
            rgb=cmap(norm(v))
            lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            win=best and v==np.nanmax(vals)
            ax.text(j,i,format(v,fmt)+(" ★" if win else ""),ha="center",va="center",
                    color="white" if lum<.48 else "#17212b",fontsize=10)
            if win:
                ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,
                             edgecolor="#e69f00",linewidth=2.5))
        setup(ax,title)
        fig.colorbar(im,ax=ax,label=unit,shrink=.85)
        save(fig,out,name)

    acc=[r for r in curves if r["method"]!="vanilla" and r["N"] in (128,256,512)]
    shared=norm_for([r["accuracy_percent"] for r in acc])
    for n in (128,256,512):
        vanilla=next(r["accuracy_percent"] for r in curves if r["N"]==n and r["method"]=="vanilla")
        heat([r for r in acc if r["N"]==n],"accuracy_percent",f"accuracy_heatmap_n{n}",
             f"N={n} accuracy | Vanilla={vanilla:.2f}%","Accuracy (%)",norm=shared,best=True,fmt=".2f")
    heat(arms,"degradation_pp","degradation_heatmap","Accuracy degradation","Percentage points",fmt=".2f")
    heat(arms,"final_flip_rate_percent","flip_rate_heatmap_n512","Selection flip rate at N=512","Percent",fmt=".2f")

    for scope in ("all","top16_original"):
        rows=[r for r in stats if r["scope"]==scope]
        c=matrix(rows,"mean_correct"); w=matrix(rows,"mean_incorrect")
        norm=norm_for(np.r_[c.ravel(),w.ravel()],bool(np.nanmin(np.r_[c.ravel(),w.ravel()])<0))
        cmap=plt.get_cmap("RdBu" if np.nanmin(np.r_[c.ravel(),w.ravel()])<0 else "Blues")
        fig,ax=plt.subplots(figsize=(max(8,1.25*len(lrs)+2),7))
        for i,j in np.ndindex(c.shape):
            for dy,v in ((0,c[i,j]),(-.48,w[i,j])):
                color=cmap(norm(v)) if np.isfinite(v) else "#eeeeee"
                ax.add_patch(Rectangle((j-.48,i+dy),.96,.48,facecolor=color,edgecolor="white"))
                lum=.2126*color[0]+.7152*color[1]+.0722*color[2] if not isinstance(color,str) else 1
                ax.text(j,i+dy+.24,f"{v:.4g}" if np.isfinite(v) else "—",
                        ha="center",va="center",fontsize=10,color="white" if lum<.48 else "#17212b")
        setup(ax,f"Mean penalty ({scope}) | upper: correct / lower: incorrect")
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,label="Reward units",shrink=.85)
        save(fig,out,f"penalty_paired_{scope}")
        heat(rows,"delta_incorrect_minus_correct",f"penalty_delta_{scope}",
             f"Incorrect − correct penalty ({scope})","Reward units",diverge=True)
        heat(rows,"relative_delta",f"relative_penalty_delta_{scope}",
             f"Relative penalty difference ({scope})","Fraction of initial gap",diverge=True)
