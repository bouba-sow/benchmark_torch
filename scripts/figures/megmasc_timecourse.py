import sys; sys.path.insert(0,"/lustre/fswork/projects/rech/ldh/una68ug/benchmark")
import glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="outputs_jeanzay/megmasc/embedding_sweep"
def psize(name):
    m=re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', name) or re.search(r'(\d+)\s*[mM]\b', name)
    if not m: return 0.0
    v=float(m.group(1)); return v*1e9 if re.search(r'[bB]', name[m.start():m.end()+1]) else v*1e6

fams={}
for f in glob.glob(f"{ROOT}/*/*/megmasc_mean_timecourse.csv"):
    fam=f.split("/")[-3]; model=f.split("/")[-2]
    d=pd.read_csv(f)
    if "time_s" not in d: continue
    if "revision" in d and d["revision"].nunique()>1: d=d[d["revision"]==d["revision"].iloc[-1]]
    fams.setdefault(fam,{})[model]=(d["time_s"].values*1000, d["pearsonr"].values)

order=["gpt2","pythia","gptneo","opt","olmo2","qwen3","llama32","mistral","mamba","mamba2"]
order=[f for f in order if f in fams]
t0=np.arange(-0.2,0.78001,0.02)*1000

# grand mean over all models
allc=[]
for fam in fams:
    for m,(t,r) in fams[fam].items(): allc.append(np.interp(t0,t,r))
grand=np.mean(allc,0)
gpk=t0[np.argmax(grand)]

fig,axes=plt.subplots(3,4,figsize=(19,12),sharex=True)
axes=axes.ravel()
for ax,fam in zip(axes,order):
    ms=sorted(fams[fam], key=lambda x:psize(x))
    cols=plt.cm.viridis(np.linspace(0,.9,len(ms)))
    fam_curves=[]
    for c,m in zip(cols,ms):
        t,r=fams[fam][m]; ri=np.interp(t0,t,r); fam_curves.append(ri)
        lbl=re.sub(r'^(openai-community_|EleutherAI_|facebook_|allenai_|Qwen_|meta-llama_|state-spaces_|AntonV_|mistralai_)','',m)
        ax.plot(t0,ri,color=c,lw=1,alpha=.85,label=lbl)
    fm=np.mean(fam_curves,0)
    ax.plot(t0,fm,color="k",lw=2.4,zorder=5)
    pk=t0[np.argmax(fm)]
    ax.axvline(pk,color="crimson",ls="--",lw=1); ax.axvline(0,color="grey",lw=.8)
    ax.axhline(0,color="grey",lw=.6)
    ax.set_title(f"{fam}   (n={len(ms)}, peak {pk:.0f} ms)",fontsize=11)
    ax.legend(fontsize=6.5,loc="upper left",ncol=1,frameon=False)
    ax.grid(alpha=.25)

# panel 11: family means overlay
ax=axes[len(order)]
for fam in order:
    fc=[np.interp(t0,*fams[fam][m][::1]) if False else np.interp(t0,fams[fam][m][0],fams[fam][m][1]) for m in fams[fam]]
    ax.plot(t0,np.mean(fc,0),lw=1.8,label=fam)
ax.axvline(0,color="grey",lw=.8);ax.axhline(0,color="grey",lw=.6);ax.grid(alpha=.25)
ax.set_title("family means overlaid",fontsize=11); ax.legend(fontsize=8,ncol=2,frameon=False)

# panel 12: grand mean + candidate windows
ax=axes[len(order)+1]
ax.plot(t0,grand,color="k",lw=2.5)
ax.axvline(0,color="grey",lw=.8);ax.axhline(0,color="grey",lw=.6);ax.grid(alpha=.25)
for w,c in [((80,180),"tab:orange"),((250,450),"tab:green"),((300,500),"tab:blue")]:
    ax.axvspan(*w,alpha=.15,color=c,label=f"{w[0]}-{w[1]} ms")
ax.axvline(gpk,color="crimson",ls="--",lw=1.5,label=f"grand peak {gpk:.0f} ms")
ax.set_title(f"GRAND MEAN over all {len(allc)} models",fontsize=11); ax.legend(fontsize=8,frameon=False)

for ax in axes: ax.set_xlim(-200,780)
for ax in axes[8:]: ax.set_xlabel("time from word onset (ms)")
for ax in axes[::4]: ax.set_ylabel("encoding Pearson r\n(best layer, mean over 208 sensors × 11 subj)")
fig.suptitle("MEG-MASC sensor-space encoding time course — best-layer r(t), per model, grouped by family",fontsize=14,y=.995)
fig.tight_layout(rect=[0,0,1,.98])
out="/tmp/claude-306950/-lustre-fswork-projects-rech-ldh-una68ug-benchmark/e646b674-92ac-48a5-af4a-43924564d103/scratchpad/megmasc_timecourse_by_family.png"
fig.savefig(out,dpi=130); print("saved",out)

# print peak table
print("\nfamily      peak_ms   r_at_peak   r@100  r@350  r@400")
for fam in order:
    fc=np.mean([np.interp(t0,fams[fam][m][0],fams[fam][m][1]) for m in fams[fam]],0)
    g=lambda ms: fc[np.argmin(np.abs(t0-ms))]
    print(f"{fam:10s}  {t0[np.argmax(fc)]:6.0f}   {fc.max():.4f}     {g(100):.4f} {g(350):.4f} {g(400):.4f}")
print(f"\nGRAND      {gpk:6.0f}   {grand.max():.4f}     {grand[np.argmin(np.abs(t0-100))]:.4f} {grand[np.argmin(np.abs(t0-350))]:.4f} {grand[np.argmin(np.abs(t0-400))]:.4f}")
