"""Validate CDF-versus-fixed-level separation for one existing training interval.

No fitting, interval, model, alternative, or tolerance is changed. Both interval
endpoints and every reported upper/lower bound share one simultaneous DKW event.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coupling as model
import numpy as np
from scipy.stats import beta, binom, chi2, ks_2samp


def read_fit():
    with (HERE/'coupling/fits.csv').open(encoding='utf-8') as stream:
        return next(r for r in csv.DictReader(stream)
                    if int(r['training_paths']) == 1024 and int(r['fit']) == 0)


def ks_details(values, reference_sorted):
    """Exact empirical KS distance, including a maximizing score and its side."""
    values = np.sort(values)
    n, m = len(values), len(reference_sorted)
    right = np.searchsorted(reference_sorted, values, side='right')/m
    left = np.searchsorted(reference_sorted, values, side='left')/m
    plus = np.arange(1,n+1)/n-right
    minus = left-np.arange(n)/n
    ip, im = int(np.argmax(plus)), int(np.argmax(minus))
    if plus[ip] >= minus[im]:
        return float(plus[ip]), float(values[ip]), float(right[ip]), 'right', float(plus[ip])
    return float(minus[im]), float(values[im]), float(left[im]), 'left', -float(minus[im])


def simulate(count, ridges, seed, batch):
    rng = np.random.default_rng(seed)
    coefficients = {ell:tuple(np.empty(count) for _ in range(3)) for ell in ridges}
    for start in range(0,count,batch):
        stop = min(start+batch,count)
        arrays = model.base.innovations(rng, stop-start, 20)
        for ell in ridges:
            part = model.coefficients(arrays,ell)
            for destination, source in zip(coefficients[ell],part):
                destination[start:stop] = source
    return coefficients


def partitions(fit):
    lo, hi = float(fit['lower']), float(fit['upper'])
    outer = (np.sqrt(1024/chi2.ppf(1-.0025/2,1024))-1,
             np.sqrt(1024/chi2.ppf(.0025/2,1024))-1)
    edges = np.linspace(*outer,9)
    covering = [(a,b) for a,b in zip(edges[:-1],edges[1:]) if a<=hi and b>=lo]
    clipped = [(max(a,lo),min(b,hi)) for a,b in covering]
    fine = list(zip(np.linspace(lo,hi,33)[:-1],np.linspace(lo,hi,33)[1:]))
    return {'original_cover':covering, 'clip_to_interval':clipped, 'split_32':fine}


def physical_check(fit):
    """Independent literal Euler evolution, with fitted zero-noise means and matrices."""
    seed, count, horizon, h = 2026092142, 64, 20, .5
    khat = .1*float(fit['fitted_ratio'])
    coefficients = {}
    arrays = model.base.innovations(np.random.default_rng(seed),count,horizon)
    for ell in (0,.03,.3):
        coefficients[ell] = model.coefficients(arrays,ell)
    worst = 0.0
    for d in (float(fit['lower']),float(fit['upper'])):
        rng = np.random.default_rng(seed)
        kappa = khat*(1+d)
        x,y = np.zeros(count),np.zeros(count)
        scores = {ell:np.zeros(count) for ell in coefficients}
        for _ in range(horizon):
            z = rng.standard_normal((3,count))
            bx,by = x.copy(),y.copy()
            covariance = np.zeros((count,2,2))
            for step in range(2):
                f = np.zeros((count,2,2))
                f[:,0,0],f[:,1,1] = .5,1.0
                f[:,1,0] = h*khat*(1-np.tanh(bx)**2)
                covariance = f@covariance@np.swapaxes(f,-1,-2)
                covariance[:,0,0] += h
                by += h*khat*np.tanh(bx)
                bx *= .5
                y += h*kappa*np.tanh(x)
                x = .5*x+np.sqrt(h)*z[step]
            residual = np.stack((x-bx,y-by),axis=1)
            for ell in scores:
                matrix = covariance.copy()
                matrix[:,1,1] += khat**2*ell
                scores[ell] += np.einsum('ni,nij,nj->n',residual,np.linalg.inv(matrix),residual)/horizon
        for ell in scores:
            expected = model.base.score_at(coefficients[ell],d)
            worst = max(worst,float(np.max(np.abs(scores[ell]-expected))))
    assert worst<1e-10
    # Check the independent empirical-CDF routine on tied, unequal-sized samples.
    rng=np.random.default_rng(1729)
    ks_error=0.0
    for _ in range(10):
        a,b=rng.integers(0,30,101),rng.integers(0,30,113)
        ks_error=max(ks_error,abs(ks_details(a,np.sort(b))[0]-ks_2samp(a,b).statistic))
    assert ks_error<1e-12
    return dict(direct_euler_score_error=worst,ks_statistic_error=ks_error)


def population_mechanism_bounds(reference, target, b, nodes=8193):
    """Monotone enclosures of the positive/negative Beta-weighted CDF areas.

    The reference and target CDFs are already in the joint DKW event.
    The finite integration mesh is bounded using monotonicity, not ignored.
    """
    u=np.linspace(0,1,nodes)
    def composed(prob, upper):
        index=np.clip(np.ceil(np.clip(prob,0,1)*len(reference)).astype(int)-1,0,len(reference)-1)
        values=np.searchsorted(target,reference[index],side='right')/len(target)
        values=np.where(prob<=0,0,np.where(prob>=1,1,values))
        return np.clip(values+(b if upper else -b),0,1)
    low_g=composed(u-b,False)
    high_g=composed(u+b,True)
    low_h=low_g[:-1]-u[1:]
    high_h=high_g[1:]-u[:-1]
    mass=np.diff(beta.cdf(u,475,25))
    positive=(float(np.dot(mass,np.maximum(low_h,0))),float(np.dot(mass,np.maximum(high_h,0))))
    negative=(float(np.dot(mass,np.maximum(-high_h,0))),float(np.dot(mass,np.maximum(-low_h,0))))
    absolute_upper=float(np.dot(mass,np.maximum(np.abs(low_h),np.abs(high_h))))
    return dict(positive_lower=positive[0],positive_upper=positive[1],
                negative_lower=negative[0],negative_upper=negative[1],
                weighted_absolute_upper=absolute_upper,integration_nodes=nodes)


def mechanism(coefficients, d, b, out):
    reference=np.sort(coefficients[0])
    target=np.sort(model.base.score_at(coefficients,d))
    n=len(reference)
    ranks=np.arange(n+1)/n
    g=binom.cdf(24,499,1-ranks)
    weights=np.diff(g)
    empirical_h=(np.searchsorted(target,reference,side='right')/n
                 -np.searchsorted(reference,reference,side='right')/n)
    signed=-float(np.dot(weights,empirical_h))
    positive=float(np.dot(weights,np.maximum(empirical_h,0)))
    negative=float(np.dot(weights,np.maximum(-empirical_h,0)))
    direct=float(np.mean(g[np.searchsorted(reference,target,side='left')])
                 -np.mean(g[np.searchsorted(reference,reference,side='left')]))
    assert abs(direct-signed)<1e-11
    band=beta.ppf([.025,.975],475,25)
    cumsum=-np.cumsum(weights*empirical_h)
    u=np.unique(np.r_[np.linspace(0,1,4001),band])
    index=np.clip(np.ceil(u*n).astype(int)-1,0,n-1)
    quantile=reference[index]
    h=np.searchsorted(target,quantile,side='right')/n-u
    h[0],h[-1]=0,0
    curve=[dict(reference_quantile=float(v),score=float(t),cdf_difference=float(delta),
                beta_density=float(beta.pdf(v,475,25)),weighted_cumulative=float(cumsum[i]))
           for v,t,delta,i in zip(u,quantile,h,index)]
    base_result=dict(witness_d=d,weighted_positive=positive,weighted_negative=negative,
        weighted_absolute=positive+negative,signed_rejection_deviation=signed,
        direct_rejection_deviation=direct,discrete_identity_error=abs(direct-signed),
        beta_central_95=band.tolist())
    base_result.update(population_mechanism_bounds(reference,target,b))
    model.base.write_csv(out/'mechanism.csv',curve)
    (out/'mechanism.json').write_text(json.dumps(base_result,indent=2),encoding='utf-8')
    return base_result


def plot(out, mechanism_result, points, bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'mechanism.csv').open(encoding='utf-8') as stream:
        curve=list(csv.DictReader(stream))
    u=np.array([float(r['reference_quantile']) for r in curve])
    h=np.array([float(r['cdf_difference']) for r in curve])
    integral=np.array([float(r['weighted_cumulative']) for r in curve])
    band=mechanism_result['beta_central_95']
    fig,axes=plt.subplots(2,2,figsize=(11,7.3))
    axes[0,0].plot(u,h,color='#28659A',lw=1.7)
    axes[0,0].axhline(0,color='#777777',lw=.7)
    axes[0,0].axhspan(-.02,.02,color='#eeeeee',alpha=.55)
    axes[0,0].axvspan(*band,color='#D27831',alpha=.22)
    axes[0,0].set_title('CDF difference at the interval endpoint',loc='left')
    axes[0,0].set_ylabel('F_d(Q_0(u)) - u')
    axes[0,0].set_xlabel('Reference quantile u')
    axes[1,0].plot(u,integral,color='#985DAD',lw=1.7)
    axes[1,0].axvspan(*band,color='#D27831',alpha=.22)
    axes[1,0].axhline(0,color='#777777',lw=.7)
    axes[1,0].set_title('Accumulated contribution to rejection error',loc='left')
    axes[1,0].set_ylabel('Negative weighted integral')
    axes[1,0].set_xlabel('Reference quantile u')
    axes[1,0].set_xlim(.85,1)
    max_n=max(int(r['paths']) for r in bounds)
    selected=[r for r in bounds if r['partition']=='split_32' and int(r['paths'])==max_n]
    x=[float(r['ridge']) for r in selected]
    lower=[max(float(p['gap_lower']) for p in points if int(p['paths'])==max_n and float(p['ridge'])==ell) for ell in x]
    upper=[float(r['cdf_upper']) for r in selected]
    axes[0,1].plot(x,lower,'o-',label='Validated lower bound',color='#28659A',ms=4)
    axes[0,1].plot(x,upper,'o-',label='Refined upper bound',color='#D27831',ms=4)
    axes[0,1].axhline(.02,ls='--',color='#777777',label='Tolerance 0.02')
    axes[0,1].set_xscale('symlog',linthresh=.003)
    axes[0,1].set_xlim(0,.35)
    axes[0,1].set_xticks([0,.003,.01,.03,.1,.3],['0','.003','.01','.03','.1','.3'])
    axes[0,1].set_xlabel('Ridge / fitted coupling squared')
    axes[0,1].set_title('Worst CDF gap over the same interval',loc='left')
    axes[0,1].legend(frameon=False,fontsize=8)
    zero=[r for r in bounds if float(r['ridge'])==0]
    styles={'original_cover':('Original covering cells','#777777'),
            'clip_to_interval':('Clip cells to actual interval','#D27831'),
            'split_32':('32 cells in actual interval','#28659A')}
    for name,(label,color) in styles.items():
        rows=[r for r in zero if r['partition']==name]
        axes[1,1].plot([int(r['paths']) for r in rows],[float(r['cdf_upper']) for r in rows],
                       'o-',label=label,color=color,ms=4)
    axes[1,1].axhline(.02,ls='--',color='#777777')
    axes[1,1].set_xscale('log',base=2)
    sample_sizes=sorted({int(r['paths']) for r in zero})
    axes[1,1].set_xticks(sample_sizes,[f'{n:,}' for n in sample_sizes])
    axes[1,1].set_xlabel('Independent validation paths')
    axes[1,1].set_title('Conservatism of the full-whitening bound',loc='left')
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',lw=.6,color='#dddddd')
        ax.tick_params(labelsize=9)
    fig.suptitle('Distribution mismatch and a fixed Monte Carlo decision',x=.075,ha='left',fontsize=15)
    fig.text(.075,.015,'Existing n=1024 fit 0; its training interval is unchanged. Orange vertical band: 95% of Beta(475,25) weight.\n'
             'CDF bounds share a 0.0025 simulation-failure budget. Curves on the left are Monte Carlo estimates.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(out/'separation.png',dpi=180)
    fig.savefig(out/'separation.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=HERE/'separation')
    parser.add_argument('--seed',type=int,default=2026092141)
    parser.add_argument('--samples',type=int,nargs='+',default=[262144,2097152])
    parser.add_argument('--ridges',type=float,nargs='+',default=[0,.001,.003,.01,.03,.1,.3])
    parser.add_argument('--batch',type=int,default=32768)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    args.samples=sorted(set(args.samples))
    args.ridges=sorted(set(args.ridges))
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        with (out/'point_bounds.csv').open(encoding='utf-8') as stream:
            points=list(csv.DictReader(stream))
        with (out/'bound_refinement.csv').open(encoding='utf-8') as stream:
            bounds=list(csv.DictReader(stream))
        plot(out,json.loads((out/'mechanism.json').read_text()),points,bounds)
        return
    start=time.perf_counter()
    fit=read_fit()
    parts=partitions(fit)
    nf=len(args.samples)*len(args.ridges)*(3+2*sum(len(c) for c in parts.values()))
    eta=.0025
    settings=dict(seed=args.seed,samples=args.samples,ridges=args.ridges,batch=args.batch,
        training_paths=1024,fit=fit,horizon=20,references=499,alpha=.05,tolerance=.02,
        training_failure=.0025,joint_simulation_failure=eta,joint_cdf_count=nf,
        partition_sizes={name:len(cells) for name,cells in parts.items()},
        exploratory_seed=2026092140)
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    checks=physical_check(fit)
    all_coefficients=simulate(max(args.samples),args.ridges,args.seed,args.batch)
    point_rows,bound_rows=[],[]
    for n in args.samples:
        b=np.sqrt(np.log(2*nf/eta)/(2*n))
        for ell in args.ridges:
            c=tuple(a[:n] for a in all_coefficients[ell])
            reference=np.sort(c[0])
            for label in ('lower','upper'):
                d=float(fit[label])
                score=model.base.score_at(c,d)
                gap,t,u,side,signed=ks_details(score,reference)
                point_rows.append(dict(paths=n,ridge=ell,endpoint=label,d=d,empirical_gap=gap,
                    gap_lower=max(0,gap-2*b),gap_upper=min(1,gap+2*b),cdf_band=b,
                    maximizing_score=t,reference_quantile=u,side=side,signed_difference=signed))
            for name,cells in parts.items():
                largest_gap,lower_p,upper_p=0.0,1.0,0.0
                for lo,hi in cells:
                    low_score,high_score=model.base.envelopes(c,(hi-lo)/2,(hi+lo)/2)
                    largest_gap=max(largest_gap,ks_details(low_score,reference)[0],ks_details(high_score,reference)[0])
                    if ell==0:
                        fl=np.searchsorted(reference,low_score,side='left')/n
                        fu=np.searchsorted(reference,high_score,side='left')/n
                        pl=max(0,float(np.mean(binom.cdf(24,499,1-np.maximum(fl-b,0))))-b)
                        pu=min(1,float(np.mean(binom.cdf(24,499,1-np.minimum(fu+b,1))))+b)
                        lower_p,upper_p=min(lower_p,pl),max(upper_p,pu)
                bound_rows.append(dict(paths=n,ridge=ell,partition=name,cells=len(cells),
                    empirical_envelope_gap=largest_gap,cdf_band=b,cdf_upper=min(1,largest_gap+2*b),
                    rejection_lower=lower_p if ell==0 else '',rejection_upper=upper_p if ell==0 else '',
                    rejection_deviation_upper=max(.05-lower_p,upper_p-.05) if ell==0 else ''))
            print(f'validation N={n:,}, ridge={ell}: endpoint gap={max(p["empirical_gap"] for p in point_rows if p["paths"]==n and p["ridge"]==ell):.6f}',flush=True)
        model.base.write_csv(out/'point_bounds.csv',point_rows)
        model.base.write_csv(out/'bound_refinement.csv',bound_rows)
    max_n=max(args.samples)
    witness=max((r for r in point_rows if r['paths']==max_n and r['ridge']==0),key=lambda r:r['gap_lower'])
    refined=next(r for r in bound_rows if r['paths']==max_n and r['ridge']==0 and r['partition']=='split_32')
    mechanism_result=mechanism(all_coefficients[0],witness['d'],witness['cdf_band'],out)
    checks.update(discrete_integral_identity_error=mechanism_result['discrete_identity_error'],
                  lower_cdf_gap=witness['gap_lower'],upper_cdf_gap=refined['cdf_upper'],
                  upper_rejection_deviation=refined['rejection_deviation_upper'],
                  separation_established=bool(witness['gap_lower']>.02 and refined['rejection_deviation_upper']<=.02))
    (out/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    plot(out,mechanism_result,point_rows,bound_rows)
    settings.update(seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__)
    import scipy
    settings['scipy']=scipy.__version__
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
