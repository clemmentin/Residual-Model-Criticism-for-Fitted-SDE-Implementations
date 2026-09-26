"""Unknown nonlinear coupling: exact training interval and paired score comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import helpers as base
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, chi2


def fit_coupling(y, failure):
    """y is Y_end/(h*kappa_true) only for generating the numerical experiment.

    In application the same equations use observed Y_end/h. The estimator
    and interval scale equivariantly, so this normalization uses no oracle
    information in the rule.
    """
    n = len(y)
    boundary = np.max(np.abs(y))
    if boundary<=0:
        raise ValueError('This experiment assumes strictly positive coupling.')
    scaled=y/boundary
    def statistic(relative_k):
        return np.sum(np.arctanh(scaled/relative_k)**2)/0.5
    def solve(target):
        left = np.nextafter(1.0, np.inf)
        right = 2.0
        while statistic(right) > target:
            right *= 2.0
        return brentq(lambda k: statistic(k)-target, left, right, xtol=1e-13)
    fitted = solve(n)
    lower = solve(chi2.ppf(1-failure/2, n))
    upper = solve(chi2.ppf(failure/2, n))
    return boundary*fitted, lower/fitted-1, upper/fitted-1


def coefficients(arrays, ridge, fitted_kappa=1.0, fixed=False):
    rx, ry, slope, variance = arrays
    x = np.zeros(len(rx))
    raw = np.empty_like(ry)
    for k in range(rx.shape[1]):
        raw[:, k] = ry[:, k] + 0.5*(np.tanh(x)+np.tanh(0.5*x))
        x = 0.25*x+rx[:, k]
    strong = np.mean(rx**2/0.625, axis=1)
    if np.isinf(ridge) and not fixed:
        return strong, np.zeros_like(strong), np.zeros_like(strong)
    residual = ry if fixed else ry-slope*rx
    precision = (np.full_like(ry, fitted_kappa**2/0.625) if fixed
                 else 1.0/(variance+ridge))
    return (strong+np.mean(residual**2*precision, axis=1),
            np.mean(residual*raw*precision, axis=1),
            np.mean(raw**2*precision, axis=1))


def select(args, out):
    arrays = base.innovations(np.random.default_rng(args.seed), args.pilot, args.horizon)
    nf = (2*args.cells+1)*len(args.ridges)*len(args.training)
    addition = 2*np.sqrt(np.log(2*nf/args.pilot_failure)/(2*args.pilot))
    b=addition/2
    rank_k=int(np.floor(.05*(args.references+1)))
    table = []
    coefs = {ridge: coefficients(arrays, ridge) for ridge in args.ridges}
    for n in args.training:
        if not chi2.ppf(args.training_failure/2,n)<n<chi2.ppf(1-args.training_failure/2,n):
            raise ValueError('The precomputed outer interval requires chi-square quantiles on both sides of n.')
        lower = np.sqrt(n/chi2.ppf(1-args.training_failure/2, n))-1
        upper = np.sqrt(n/chi2.ppf(args.training_failure/2, n))-1
        edges = np.linspace(lower, upper, args.cells+1)
        for ridge, c in coefs.items():
            reference=np.sort(c[0])
            for lo, hi in zip(edges[:-1], edges[1:]):
                low_score, high_score = base.envelopes(c, (hi-lo)/2, (hi+lo)/2)
                gap = max(base.cdf_gap(low_score, c[0]), base.cdf_gap(c[0], high_score))
                fl=np.searchsorted(reference,low_score,side='left')/args.pilot
                fu=np.searchsorted(reference,high_score,side='left')/args.pilot
                lower_rejection=max(0.0,float(np.mean(binom.cdf(rank_k-1,args.references,
                                                        1-np.maximum(fl-b,0))))-b)
                upper_rejection=min(1.0,float(np.mean(binom.cdf(rank_k-1,args.references,
                                                        1-np.minimum(fu+b,1))))+b)
                table.append(dict(training_paths=n, ridge=ridge, lower=lo, upper=hi,
                                  empirical_gap=gap, dkw_addition=addition,
                                  bound=min(1.0, gap+addition),
                                  lower_rejection=lower_rejection,upper_rejection=upper_rejection))
        print(f"coupling pilot n={n}: relative interval [{lower:.4g}, {upper:.4g}]", flush=True)
    base.write_csv(out/'cell_bounds.csv', table)


def evaluate(args, out):
    with (out/'cell_bounds.csv').open(encoding='utf-8') as stream:
        table = list(csv.DictReader(stream))
    train_rng = np.random.default_rng(args.seed+10)
    fits = []
    for n in args.training:
        for fit in range(args.fits):
            observation = np.tanh(np.sqrt(0.5)*train_rng.standard_normal(n))
            khat, lower, upper = fit_coupling(observation, args.training_failure)
            selected, bound, level_ridge, level_bound = np.inf, 0.0, np.inf, 0.0
            target=np.floor(.05*(args.references+1))/(args.references+1)
            for ridge in args.ridges:
                relevant = [r for r in table if int(r['training_paths'])==n
                          and float(r['ridge'])==ridge and float(r['lower'])<=upper
                          and float(r['upper'])>=lower]
                worst = max(float(r['bound']) for r in relevant)
                if worst <= args.gap_target and np.isinf(selected):
                    selected, bound = ridge, worst
                level_worst=max(max(target-float(r['lower_rejection']),
                                    float(r['upper_rejection'])-target,0.0) for r in relevant)
                if level_worst<=args.gap_target and np.isinf(level_ridge):
                    level_ridge,level_bound=ridge,level_worst
            fits.append(dict(training_paths=n, fit=fit, fitted_ratio=khat,
                             lower=lower, upper=upper, error=1/khat-1,
                             selected_ridge=selected, selected_bound=bound,
                             level_ridge=level_ridge,level_bound=level_bound,
                             covers=bool(lower<=1/khat-1<=upper)))
    base.write_csv(out/'fits.csv', fits)
    specifications = {'reference':{}, 'null':{}, 'calibration':{},
                      'strong':{'strong':args.strong}, 'weak':{'weak':args.weak}}
    banks = {name:base.innovations(np.random.default_rng(args.seed+20+i), args.paths,
                                   args.horizon, **spec)
             for i,(name,spec) in enumerate(specifications.items())}
    # In fixed raw coordinates, only this scalar factor changes with the fit.
    fixed_templates = {name:coefficients(bank, 0, 1, True) for name,bank in banks.items()}
    strong_scores = {name:np.mean(bank[0]**2/0.625, axis=1) for name,bank in banks.items()}
    cache, rows = {}, []
    target = np.floor(0.05*(args.references+1))/(args.references+1)
    for kappa in args.kappas:
        for n in args.training:
            for fit in [r for r in fits if r['training_paths']==n]:
                error, khat = fit['error'], kappa*fit['fitted_ratio']
                for method in ('full','selected','level','fixed'):
                    ridge = (fit['selected_ridge'] if method=='selected' else
                             fit['level_ridge'] if method=='level' else 0.0)
                    if method=='fixed':
                        coefs = {name:(strong_scores[name]+khat**2*(c[0]-strong_scores[name]),
                                      khat**2*c[1], khat**2*c[2])
                                 for name,c in fixed_templates.items()}
                    else:
                        if ridge not in cache:
                            cache[ridge] = {name:coefficients(bank,ridge) for name,bank in banks.items()}
                        coefs = cache[ridge]
                    reference = coefs['reference'][0]
                    scores = {name:base.score_at(c,error) for name,c in coefs.items() if name!='reference'}
                    null = base.rank_rejection(scores['null'],reference,args.references)
                    oracle = base.rank_rejection(coefs['null'][0],reference,args.references)
                    rows.append(dict(kappa=kappa,training_paths=n,fit=fit['fit'],method=method,
                        fitted_kappa=khat,relative_error=error,ridge_over_fitted_kappa_squared=ridge,
                        gap=base.cdf_gap(scores['null'],coefs['null'][0]),
                        independent_gap=base.cdf_gap(scores['null'],reference),
                        null_rejection=null,oracle_null=oracle,
                        null_rejection_cv=float(target+null-oracle),
                        matched_null=base.rank_rejection(scores['null'],scores['calibration'],args.references),
                        strong_raw=base.rank_rejection(scores['strong'],reference,args.references),
                        strong_matched=base.rank_rejection(scores['strong'],scores['calibration'],args.references),
                        weak_raw=base.rank_rejection(scores['weak'],reference,args.references),
                        weak_matched=base.rank_rejection(scores['weak'],scores['calibration'],args.references)))
            print(f"coupling evaluated kappa={kappa}, n={n}",flush=True)
    base.write_csv(out/'results.csv',rows)
    summary=[]
    for kappa in args.kappas:
        for n in args.training:
            for method in ('full','selected','level','fixed'):
                group=[r for r in rows if (r['kappa'],r['training_paths'],r['method'])==(kappa,n,method)]
                r=dict(kappa=kappa,training_paths=n,method=method)
                for key in ('ridge_over_fitted_kappa_squared','gap','independent_gap','null_rejection',
                            'oracle_null','null_rejection_cv','matched_null','strong_raw',
                            'strong_matched','weak_raw','weak_matched'):
                    values=np.array([v[key] for v in group])
                    r[key],r[key+'_min'],r[key+'_max']=float(values.mean()),float(values.min()),float(values.max())
                summary.append(r)
    base.write_csv(out/'summary.csv',summary)
    base.plot_results(out)


def checks():
    rng=np.random.default_rng(281009)
    y=np.tanh(np.sqrt(.5)*rng.standard_normal(100))
    fit,lo,hi=fit_coupling(y,.0025)
    pivot_error=max(abs(np.sum(np.arctanh(y/(fit*(1+lo)))**2)/.5-chi2.ppf(1-.0025/2,100)),
                    abs(np.sum(np.arctanh(y/(fit*(1+hi)))**2)/.5-chi2.ppf(.0025/2,100)))
    assert pivot_error<1e-8
    fit2,lo2,hi2=fit_coupling(.03*y,.0025)
    equivariance=max(abs(fit2/.03-fit),abs(lo2-lo),abs(hi2-hi))
    assert equivariance<1e-10
    n=len(y)
    outer=(np.sqrt(n/chi2.ppf(1-.0025/2,n))-1,np.sqrt(n/chi2.ppf(.0025/2,n))-1)
    assert outer[0]<=lo<=hi<=outer[1]
    # Direct physical residual calculation at a deliberately wrong fitted coupling.
    arrays=base.innovations(rng,31,13)
    rx,ry,slope,variance=arrays
    x=np.zeros(31)
    raw=np.empty_like(ry)
    for k in range(13):
        raw[:,k]=ry[:,k]+.5*(np.tanh(x)+np.tanh(.5*x))
        x=.25*x+rx[:,k]
    max_error=0.0
    for khat in (.07,.3,1.7):
        actual_kappa=.4
        fitted_residual_y=khat*ry+(actual_kappa-khat)*raw
        for ridge in (0,.03,.3):
            direct=np.mean(rx**2/.625+(fitted_residual_y-khat*slope*rx)**2
                           /(khat**2*(variance+ridge)),axis=1)
            polynomial=base.score_at(coefficients(arrays,ridge),actual_kappa/khat-1)
            max_error=max(max_error,float(np.max(np.abs(direct-polynomial))))
    assert max_error<1e-10
    u=(np.arange(10000)+.5)/10000
    rank_error=abs(float(np.mean(binom.cdf(24,499,1-u)))-.05)
    assert rank_error<1e-10
    return dict(scale_equivariance_error=equivariance,score_polynomial_max_error=max_error,
                confidence_endpoint_pivot_error=pivot_error,finite_rank_integration_error=rank_error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=Path(__file__).parent/'coupling')
    p.add_argument('--pilot',type=int,default=262144)
    p.add_argument('--paths',type=int,default=32768)
    p.add_argument('--fits',type=int,default=64)
    p.add_argument('--horizon',type=int,default=20)
    p.add_argument('--training',type=int,nargs='+',default=[64,1024,16384])
    p.add_argument('--kappas',type=float,nargs='+',default=[.1,1.0])
    p.add_argument('--ridges',type=float,nargs='+',default=[0,.001,.003,.01,.03,.1,.3,1,3,10])
    p.add_argument('--cells',type=int,default=8)
    p.add_argument('--training-failure',type=float,default=.0025)
    p.add_argument('--pilot-failure',type=float,default=.0025)
    p.add_argument('--gap-target',type=float,default=.02)
    p.add_argument('--references',type=int,default=499)
    p.add_argument('--strong',type=float,default=1.15)
    p.add_argument('--weak',type=float,default=.2)
    p.add_argument('--seed',type=int,default=2026092121)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    args=p.parse_args()
    args.ridges=sorted(set(args.ridges))
    start=time.perf_counter()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        base.plot_results(args.out_dir)
        return
    (args.out_dir/'checks.json').write_text(json.dumps(checks(),indent=2),encoding='utf-8')
    if not args.evaluate_only:
        select(args,args.out_dir)
    evaluate(args,args.out_dir)
    settings=vars(args).copy()
    settings['out_dir']=str(args.out_dir)
    settings['seconds']=time.perf_counter()-start
    settings['numpy']=np.__version__
    import scipy
    settings['scipy']=scipy.__version__
    settings['python']=sys.version
    (args.out_dir/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
