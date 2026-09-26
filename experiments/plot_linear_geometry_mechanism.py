"""Validate retained ranks and render the design-fixed linear mechanism figures."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


from experiments.run_linear_geometry_mechanism import OUT, ROOT, euler_covariance

FIG = ROOT / 'output/figures'


def audit() -> dict:
    required = {
        'trials.csv': ('geometry_index', 'alternative', 'replicate',
                       'p_instantaneous', 'p_finite_step'),
        'summary.csv': ('geometry_index', 'alternative', 'n',
                        'rate_instantaneous', 'rate_finite_step'),
    }
    shapes = {}
    for name, columns in required.items():
        path = OUT / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing linear geometry output: {path}')
        frame = pd.read_csv(path, nrows=0)
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f'{path} is missing columns: {missing}')
        shapes[name] = len(pd.read_csv(path, usecols=[columns[0]]))
    return {'retained_trial_rows': shapes['trials.csv'],
            'summary_rows': shapes['summary.csv']}


def contour(ax, cov, color, label, style='-'):
    values,vectors = np.linalg.eigh(cov)
    if values[0] < 1e-14:
        vec = vectors[:,1]*np.sqrt(values[1])
        ax.plot([-vec[0],vec[0]],[-vec[1],vec[1]],color=color,lw=2.2,label=label,ls=style)
    else:
        angle = np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))
        patch = Ellipse((0,0),2*np.sqrt(values[1]),2*np.sqrt(values[0]),angle=angle,
                        facecolor='none',edgecolor=color,lw=2,ls=style,label=label)
        ax.add_patch(patch)


def geometry_figure():
    fig,axes = plt.subplots(1,3,figsize=(8.4,2.8),layout='constrained')
    for i,s2 in enumerate([0.,1.]):
        ax=axes[i]
        contour(ax,np.diag([1.,s2*s2]),'#D17A22','Instantaneous / L=1','--')
        contour(ax,euler_covariance(1,10,1,1,s2),'#235A91','Euler L=10')
        ax.axhline(0,color='#CAD2DA',lw=.5,zorder=0)
        ax.axvline(0,color='#CAD2DA',lw=.5,zorder=0)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect='equal',xlabel=r'$X_1$ residual',ylabel=r'$X_2$ residual')
        ax.set_title(['(a) Support emergence', '(b) Scale and covariance'][i],fontsize=10,fontweight='semibold')
        ax.text(.03,.94,rf'$\sigma_2={s2:g}$',transform=ax.transAxes,va='top',fontsize=9)
        ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    axes[0].legend(loc='lower left',fontsize=7,frameon=False,handlelength=2.0)
    ax=axes[2]; L=np.arange(1,21)
    b=(L-1)*(2*L-1)/(6*L*L)
    for eta,color in zip([.5,1.5,3],['#7D9A52','#235A91','#993D4F']):
        ax.plot(L,eta*eta*b,color=color,lw=1.8,label=rf'$\eta={eta:g}$')
    ax.set(xlabel='Euler substeps L',ylabel='Propagated / direct variance',xlim=(1,20),ylim=(0,3.2))
    ax.set_xticks([1,5,10,20]); ax.set_title('(c) A continuous scale effect',fontsize=10,fontweight='semibold')
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.grid(axis='y',color='#DDE2E8',lw=.5)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for ext in ['pdf','png']:
        fig.savefig(FIG/f'paper1_finite_grid_geometry.{ext}',dpi=220)
    plt.close(fig)


def sensitivity_figure():
    summary=pd.read_csv(OUT/'summary.csv',keep_default_na=False)
    fig,axes=plt.subplots(2,3,figsize=(7.6,5.4),layout='constrained')
    cmap=plt.get_cmap('RdBu_r'); norm=Normalize(-20,20)
    symbols={'equivalent':'E','materially_higher':'H','materially_lower':'L','unresolved':'.'}
    for row,strength in enumerate(['1p10','1p25']):
        for col,family in enumerate(['driver1','driver2','uniform']):
            ax=axes[row,col]; alt=f'{family}_{strength}'
            data=summary[summary.alternative.eq(alt)]
            values=data.pivot(index='kappa',columns='L',values='difference')*100
            status=data.pivot(index='kappa',columns='L',values='classification')
            ax.imshow(values.to_numpy(),cmap=cmap,norm=norm,aspect='auto')
            for y in range(4):
                for x in range(4):
                    value=values.iloc[y,x]
                    label=f'{value:.1f}\n{symbols[status.iloc[y,x]]}'
                    ax.text(x,y,label,ha='center',va='center',fontsize=8.4,
                            color='white' if abs(value)>=12 else '#172637',linespacing=1.1)
            ax.set_xticks(range(4),[1,2,5,10]); ax.set_yticks(range(4),['0','0.5','1.5','3'])
            ax.set_xlabel('Euler substeps L',fontsize=8.5)
            ax.set_ylabel(r'Coupling $\kappa$',fontsize=8.5)
            name={'driver1':'Driver 1','driver2':'Driver 2','uniform':'Both drivers'}[family]
            multiplier='1.10' if row==0 else '1.25'
            ax.set_title(f'({chr(97+3*row+col)}) {name} x {multiplier}',fontsize=9.2,fontweight='semibold')
            ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
            ax.grid(which='minor',color='white',lw=.6); ax.tick_params(which='minor',bottom=False,left=False)
            for spine in ax.spines.values(): spine.set_visible(False)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,location='bottom',shrink=.72,aspect=36,pad=.02)
    bar.set_label('Observation-step minus instantaneous rejection rate (percentage points)',fontsize=8.5)
    bar.set_ticks([-20,-10,0,10,20])
    for ext in ['pdf','png']:
        fig.savefig(FIG/f'paper1_linear_coordinate_sensitivity.{ext}',dpi=220)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
    report=audit()
    geometry_figure(); sensitivity_figure()
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
