from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]
NAVY = '#1F4E79'; TEAL = '#00857C'; AMBER = '#D99000'; GRAY = '#6B778A'; LTGRAY='#F3F5F7'; LTBLUE='#EAF1F7'; LTTEAL='#E6F4F2'; LTAMBER='#FFF4DF'; DARK='#20252B'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 10.5,
    'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.edgecolor': '#6E7781', 'axes.linewidth': .7,
    'grid.color': '#DDE2E7', 'grid.linewidth': .6, 'grid.alpha': .9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

def clean(ax, grid=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if grid: ax.grid(axis='y', zorder=0)
    ax.tick_params(color='#6E7781', labelcolor='#3F4954')
    ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(OUT/name, bbox_inches='tight', pad_inches=.035, facecolor='white')
    plt.close(fig)

def box(ax, xy, w,h, text, edge=NAVY, face='white', fs=8, weight='normal'):
    p=FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.015,rounding_size=0.015',facecolor=face,edgecolor=edge,linewidth=1.15,transform=ax.transAxes,clip_on=False)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fs,fontweight=weight,transform=ax.transAxes,color=DARK)
    return p

def arrow(ax, a,b,color=GRAY):
    ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',textcoords='axes fraction',arrowprops=dict(arrowstyle='-|>',lw=1,color=color,shrinkA=2,shrinkB=2))

# Fig 4 theory-evidence bridge
fig,axs=plt.subplots(1,3,figsize=(11.6,3.2),gridspec_kw={'width_ratios':[1.18,1,1]})
ax=axs[0]; ax.axis('off'); ax.set_title('(a) Identifiability and stability',fontweight='bold',loc='left')
box(ax,(.10,.70),.80,.16,r'Active design\n$H_Q=\lambda I+\sum_p G_p^\top G_p$',edge=NAVY,face=LTBLUE,fs=8.2); arrow(ax,(.50,.69),(.50,.58))
box(ax,(.10,.48),.80,.14,r'$\sigma_{min}(A_Q)\uparrow$   and   $\kappa(A_Q)\downarrow$',edge=NAVY,face='white',fs=8.2); arrow(ax,(.50,.47),(.50,.36))
box(ax,(.10,.24),.80,.15,r'$\|\hat{Y}-Y^*\|_2\leq 2\|\eta\|_2/\sigma_{min}(A_Q)$',edge=TEAL,face=LTTEAL,fs=8.2)
ax.text(.5,.13,r'Support margin: $\gamma_{min}>2\varepsilon_Q\Rightarrow \hat{T}=T$',ha='center',va='center',transform=ax.transAxes,color=GRAY,fontsize=7.7)
ax.text(.5,.045,r'Confirmatory: $\sigma_{min}$ 0.514→0.596;  $\kappa$ 3.266→2.713',ha='center',va='center',transform=ax.transAxes,color=GRAY,fontsize=7.3)
ax=axs[1]; clean(ax); ax.set_title('(b) Support recovery',fontweight='bold',loc='left'); ax.set_ylabel('Parent-F1'); ax.set_ylim(.69,.90); ax.set_xticks([0,1],['L1 linear','L2 TIES'])
# draw left=random right=active within each x
for x,(fr,fa,ar,aa) in enumerate([(.702,.711,.747,.778),(.819,.857,.865,.867)]):
    ax.plot([x-.13,x+.13],[fr,fa],'-o',color=TEAL,lw=1.8,ms=5,label='FAS' if x==0 else None)
    ax.plot([x-.13,x+.13],[ar,aa],'-s',color=NAVY,lw=1.5,ms=4.6,label='Absolute' if x==0 else None)
ax.legend(frameon=False,loc='upper left'); ax.text(.02,.02,'left = Random   right = Active',transform=ax.transAxes,color=GRAY,fontsize=7.3)
ax=axs[2]; clean(ax); ax.set_title('(c) False-parent leakage',fontweight='bold',loc='left'); ax.set_ylabel('False-parent leakage'); ax.set_ylim(.14,.28); x=np.arange(2); w=.32
r=[.2611,.2171]; a=[.2306,.1674]; ax.bar(x-w/2,r,w,color='#AEB5BD',label='Random',zorder=3); ax.bar(x+w/2,a,w,color=TEAL,label='Active',zorder=3); ax.set_xticks(x,['L1','L2']); ax.legend(frameon=False)
for xx,v in zip(x-w/2,r): ax.text(xx,v+.004,f'{v:.3f}',ha='center',fontsize=7.2,color=GRAY)
for xx,v in zip(x+w/2,a): ax.text(xx,v+.004,f'{v:.3f}',ha='center',fontsize=7.2,color=TEAL)
fig.subplots_adjust(wspace=.42)
save(fig,'fig4_theory_evidence_bridge.pdf')

# Fig 5 selective phase
fig,axs=plt.subplots(1,2,figsize=(11.6,3.05),gridspec_kw={'width_ratios':[1,1.35]})
ax=axs[0]; ax.axis('off'); ax.set_title('(a) Selective calibration',fontweight='bold',loc='left')
box(ax,(.20,.66),.60,.15,'Signal calibration\nbase-like null → $p_{sig}$',edge=NAVY,face=LTBLUE,fs=8.7); arrow(ax,(.50,.65),(.50,.53))
box(ax,(.20,.42),.60,.15,'Bank calibration\nbank-complete null → $p_{open}$',edge=TEAL,face=LTTEAL,fs=8.7); arrow(ax,(.50,.41),(.50,.30))
box(ax,(.10,.17),.80,.12,'Low-Signal    |    Bank-Insufficient    |    Decomposable',edge=GRAY,face='white',fs=8.1,weight='bold')
ax=axs[1]; clean(ax,grid=False); ax.set_title('(b) Selective-state phase diagram',fontweight='bold',loc='left'); ax.set_xlim(0,.12); ax.set_ylim(0,.75); ax.set_xlabel(r'$p_{sig}$'); ax.set_ylabel(r'$p_{open}$')
ax.axvspan(.05,.12,color=LTAMBER,zorder=0); ax.axvspan(0,.05,color=LTTEAL,zorder=0); ax.axhspan(0,.05,xmin=0,xmax=.05/.12,color='#FBEAEA',zorder=1)
ax.axvline(.05,color='#9AA4AE',ls='--',lw=.8); ax.axhline(.05,color='#9AA4AE',ls='--',lw=.8)
ps=.02439; g0=[(.0732,'23'),(.4390,'47'),(.1951,'71')]
for po,l in g0: ax.scatter(ps,po,s=36,color=NAVY,zorder=5); ax.text(ps+.002,po+.01,l,fontsize=7,color=GRAY)
for po in [.6585,.5610]: ax.scatter(ps,po,s=42,facecolors='none',edgecolors=TEAL,marker='^',lw=1.3,zorder=5)
ax.text(.078,.38,'Low-Signal',color=AMBER,fontweight='bold',fontsize=10)
ax.text(.016,.56,'Decomposable',color=TEAL,fontweight='bold',fontsize=10,rotation=90)
ax.text(.014,.018,'Bank-\nInsufficient',color='#B94A48',fontsize=7,ha='center',va='center')
ax.legend(['matched $G_0$','TIES diagnostic'],frameon=False,loc='upper right')
fig.subplots_adjust(wspace=.35)
save(fig,'fig5_selective_phase.pdf')

# Fig 6 functional validity
fig,axs=plt.subplots(1,3,figsize=(11.6,3.25),gridspec_kw={'width_ratios':[.9,1.15,1.2]})
ax=axs[0]; ax.axis('off'); ax.set_title('(a) Exact counterfactual game',fontweight='bold',loc='left')
box(ax,(.10,.67),.80,.16,'3-parent fixed-coefficient game\nall $2^3$ subsets',edge=NAVY,face=LTBLUE,fs=8.7); arrow(ax,(.5,.66),(.5,.53)); box(ax,(.10,.43),.80,.15,'Held-out utility $U_D(M_S)$\n96 items / domain',edge=GRAY,face='white',fs=8.7); arrow(ax,(.5,.42),(.5,.29)); box(ax,(.10,.19),.80,.15,r'Exact Shapley $\varphi_i(D)$' + '\n+ leave-one-parent-out',edge=TEAL,face=LTTEAL,fs=8.7); ax.text(.5,.07,r'Efficiency residual ≤ $2.8\times10^{-17}$',transform=ax.transAxes,ha='center',color=GRAY,fontsize=7.4)
ax=axs[1]; clean(ax); ax.set_title('(b) Functional alignment',fontweight='bold',loc='left'); ax.set_ylabel(r'Mean Spearman $\rho$'); ax.set_ylim(0,.65); x=np.arange(2); w=.22
vals={'Construction':[.111,.111],'Absolute':[.596,.207],'FAS':[.556,.444]}; cols={'Construction':'#AEB5BD','Absolute':NAVY,'FAS':TEAL}
for i,(k,v) in enumerate(vals.items()): ax.bar(x+(i-1)*w,v,w,color=cols[k],label=k,zorder=3); [ax.text(xx+(i-1)*w,y+.015,f'{y:.3f}',ha='center',fontsize=7) for xx,y in zip(x,v)]
ax.set_xticks(x,['Exact Shapley','LOPO']); ax.legend(frameon=False,loc='upper right'); ax.text(.03,.03,'FAS − construction on Shapley = +0.444',transform=ax.transAxes,color=TEAL,fontweight='bold',fontsize=7.5)
ax=axs[2]; clean(ax,grid=False); ax.set_title('(c) Seed × domain consistency',fontweight='bold',loc='left'); mat=np.array([[.5,0,.5],[1,0,0],[2,0,0]],float); im=ax.imshow(mat,cmap='BuGn',vmin=0,vmax=2,aspect='auto'); ax.set_xticks(range(3),['Math','Medical','Science']); ax.set_yticks(range(3),['23','47','71']); ax.set_ylabel('Seed')
for i in range(3):
    for j in range(3): ax.text(j,i,f'+{mat[i,j]:.1f}' if mat[i,j]>0 else '0',ha='center',va='center',color='white' if mat[i,j]>1.2 else DARK,fontsize=8)
ax.text(.5,-.18,'FAS ≥ construction in 9/9 cells',transform=ax.transAxes,ha='center',color=TEAL,fontweight='bold',fontsize=8)
fig.subplots_adjust(wspace=.38)
save(fig,'fig6_functional_validity.pdf')

# Fig 7 robust audit
fig,axs=plt.subplots(1,3,figsize=(11.6,3.15),gridspec_kw={'width_ratios':[1.05,.9,.95]})
ax=axs[0]; clean(ax); ax.set_title('(a) Evaluator invariance',fontweight='bold',loc='left'); enc=['MPNet','MiniLM','BGE']; y=np.arange(3); ax.scatter([1,1,1],y,s=36,facecolors='none',edgecolors=TEAL,lw=1.4); ax.set_yticks(y,enc); ax.invert_yaxis(); ax.set_xlim(.94,1.01); ax.set_xlabel('Agreement with primary evaluator'); ax.set_xticks([.95,.975,1.0]); ax.text(.22,.5,'●  Support Jaccard / coord. ρ = 1.0',transform=ax.transAxes,color=DARK,fontsize=7.6)
for yi,r in zip(y,[.654,.656,.643]): ax.text(.03,yi,f'resid. {r:.3f}',transform=ax.get_yaxis_transform(),ha='left',va='center',color=GRAY,fontsize=7)
ax=axs[1]; clean(ax); ax.set_title('(b) Restricted-bank stress',fontweight='bold',loc='left'); vals=[0,.524,.746]; labels=['Known false\nBI','Withheld\nBI','Open-set\nAUROC']; colors=['#AEB5BD',TEAL,AMBER]; b=ax.bar(range(3),vals,color=colors,zorder=3); ax.set_ylim(0,.82); ax.set_ylabel('Rate / AUROC'); ax.set_xticks(range(3),labels); [ax.text(i,v+.025,f'{v:.3f}',ha='center',fontweight='bold',fontsize=7.5) for i,v in enumerate(vals)]
ax=axs[2]; clean(ax); ax.set_title('(c) Structural controls',fontweight='bold',loc='left'); vals=[.654,.866]; b=ax.bar([0,1],vals,color=[NAVY,TEAL],zorder=3); ax.set_xticks([0,1],['Correct pairing','Pair shuffle']); ax.set_ylabel('Mean residual ↓'); ax.set_ylim(.55,.92); ax.text(.5,.785,'+0.212',ha='center',color=TEAL,fontsize=8); ax.annotate('',xy=(1,.84),xytext=(.58,.81),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1)); [ax.text(i,v+.015,f'{v:.3f}',ha='center',fontweight='bold',fontsize=7.5) for i,v in enumerate(vals)]; ax.text(.02,.05,'Pair-shuffle collapse: 3/3 seeds\nBase-zero false signal: 0/3',transform=ax.transAxes,color=GRAY,fontsize=7.5)
fig.subplots_adjust(wspace=.40)
save(fig,'fig7_robust_audit.pdf')

# Fig 8 deep ancestry
fig,axs=plt.subplots(1,3,figsize=(11.6,3.05)); stages=['D1\nMerge','D2\nSFT','D3\nINT4','D4\nKD']; x=np.arange(4)
series={
 'FAS Active':(TEAL,'o','-',[.79365,.85714,.85714,.79365],[.75010,.78435,.78609,.74533],[.58462,.65762,.70459,.71056]),
 'FAS Random':(GRAY,'o','--',[.85714,.85714,.85714,.83810],[.77491,.81676,.80799,.82604],[.58533,.64955,.69364,.73512]),
 'Absolute Active':(NAVY,'s','-',[.90476,.90476,.85714,.85714],[.81861,.82987,.83062,.70384],[.56558,.73625,.77978,.76141]),
}
for ax,title,idx,ylab,ylim in zip(axs,['(a) Support persistence','(b) True-parent mass','(c) Residual'],[3,4,5],['Parent-F1','True-parent mass','Relative residual ↓'],[(.74,.93),(.68,.86),(.54,.81)]):
    clean(ax); ax.set_title(title,fontweight='bold',loc='left'); ax.set_ylabel(ylab); ax.set_xticks(x,stages); ax.set_ylim(*ylim)
    for name,(c,m,ls,*vals) in series.items(): ax.plot(x,vals[idx-3],ls,marker=m,color=c,lw=1.7,ms=4.7,label=name)
axs[0].text(.03,.04,'D4 / D1 F1 retention ≈ 100%',transform=axs[0].transAxes,color=TEAL,fontweight='bold',fontsize=7.6)
axs[1].text(.08,.04,'D4 FAS-A mass = 0.745',transform=axs[1].transAxes,color=TEAL,fontweight='bold',fontsize=7.6)
axs[2].text(.02,.04,'D4 FAS-A 0.711 vs Absolute-A 0.761',transform=axs[2].transAxes,color=TEAL,fontweight='bold',fontsize=7.4)
handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=3,frameon=False)
fig.subplots_adjust(wspace=.38,top=.78)
save(fig,'fig8_deep_ancestry.pdf')

# Fig 9 cross-scale transport
fig,axs=plt.subplots(1,3,figsize=(11.6,3.05));
ax=axs[0]; clean(ax); ax.set_title('(a) Cross-scale support',fontweight='bold',loc='left'); labels=['FAS-A','FAS-R','Abs-A','Abs-R','DNA']; vals=[0,0,.5778,.7937,.2667]; cols=[TEAL,'#73B8AA',NAVY,'#7292B0','#6F737A']; ax.bar(range(5),vals,color=cols,zorder=3); ax.set_xticks(range(5),labels); ax.set_ylabel('Parent-F1'); ax.set_ylim(0,.86); [ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=7) for i,v in enumerate(vals)]; ax.text(.5,-.18,'4B → 1.7B Mixture-KD',transform=ax.transAxes,ha='center',color=GRAY,fontsize=7.4)
ax=axs[1]; clean(ax); ax.set_title('(b) Scale-matched geometry',fontweight='bold',loc='left'); vals=[.95784,.67443]; ax.bar([0,1],vals,color=[AMBER,TEAL],zorder=3); ax.set_xticks([0,1],['Cross-scale','Same-scale']); ax.set_ylabel('Pooled residual ↓'); ax.set_ylim(.60,1.0); [ax.text(i,v+.012,f'{v:.3f}',ha='center',fontweight='bold',fontsize=7.5) for i,v in enumerate(vals)]; ax.annotate('−0.283',xy=(1,.72),xytext=(.35,.80),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1),color=TEAL,fontsize=8)
ax=axs[2]; clean(ax); ax.set_title('(c) LOSO transport',fontweight='bold',loc='left'); xx=[0,1,2]; vv=[.95784,.95710,.67443]; ax.plot(xx[:2],vv[:2],'-o',color=NAVY,lw=1.6,ms=4.5); ax.scatter([2],[vv[2]],color=TEAL,s=35,zorder=3); ax.set_xticks(xx,['Raw cross-scale','LOSO bridge','Same-scale\nreference']); ax.set_ylabel('Residual ↓'); ax.set_ylim(.63,1.0); [ax.text(i,v+.012,f'{v:.3f}',ha='center',fontsize=7) for i,v in enumerate(vv)]; ax.text(.05,.10,'Oracle-gap recovery: 0.26%\nΔ field cosine: −0.0155',transform=ax.transAxes,color=GRAY,fontsize=7.5)
fig.subplots_adjust(wspace=.42)
save(fig,'fig9_crossscale_transport.pdf')

print('Generated', ', '.join(str(p.name) for p in sorted(OUT.glob('fig[4-9]_*.pdf'))))
