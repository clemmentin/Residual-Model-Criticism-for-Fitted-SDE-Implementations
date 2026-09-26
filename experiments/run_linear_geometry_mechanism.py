"""Run the paired coordinate sensitivity experiment under an exact Euler law.

The design file supplies the fixed experiment settings. The simulator uses
exact Gaussian residual scatter sufficient statistics, not approximate
chi-square tails.
"""
from __future__ import annotations


import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'docs/results/linear_geometry_design.json'
OUT = ROOT / 'cache/summaries/linear_geometry_mechanism/frozen'


def euler_covariance(H: float, L: int, kappa: float, sigma1: float, sigma2: float) -> np.ndarray:
    if H <= 0 or L < 1 or int(L) != L or min(sigma1, sigma2) < 0:
        raise ValueError('Invalid linear geometry parameters.')
    a = (L - 1) / (2 * L)
    b = (L - 1) * (2 * L - 1) / (6 * L * L)
    return np.array([[sigma1**2 * H, kappa * sigma1**2 * H**2 * a],
                     [kappa * sigma1**2 * H**2 * a,
                      sigma2**2 * H + kappa**2 * sigma1**2 * H**3 * b]], dtype=float)


def continuous_covariance(H: float, kappa: float, sigma1: float, sigma2: float) -> np.ndarray:
    return np.array([[sigma1**2 * H, kappa * sigma1**2 * H**2 / 2],
                     [kappa * sigma1**2 * H**2 / 2,
                      sigma2**2 * H + kappa**2 * sigma1**2 * H**3 / 3]])


def cholesky2(cov: np.ndarray) -> np.ndarray:
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
        raise ValueError('Covariance must be finite and symmetric.')
    a = float(cov[0, 0])
    if a <= 0:
        raise ValueError('Non-positive first Cholesky pivot; no regularization allowed.')
    l11 = math.sqrt(a)
    l21 = float(cov[1, 0]) / l11
    pivot = float(cov[1, 1]) - l21 * l21
    if pivot <= 0:
        raise ValueError('Non-positive second Cholesky pivot; no regularization allowed.')
    return np.array([[l11, 0.0], [l21, math.sqrt(pivot)]])


def score_weights(generating_covariance: np.ndarray, coordinate_covariance: np.ndarray) -> np.ndarray:
    """Trace weights for r=C*u and whitening by the fixed coordinate covariance."""
    c = cholesky2(generating_covariance)
    w = cholesky2(coordinate_covariance)
    # Solve explicitly to make identical maps in the control cells bitwise equal.
    a00 = c[0, 0] / w[0, 0]
    a10 = (c[1, 0] - w[1, 0] * a00) / w[1, 1]
    a11 = c[1, 1] / w[1, 1]
    return np.array([a00*a00 + a10*a10, 2*a10*a11, a11*a11])


def scatter_draws(rng: np.random.Generator, shape: tuple[int, ...], K: int) -> np.ndarray:
    """Independent Wishart(2,K,I) draws returned as (W11,W12,W22)."""
    if K < 2:
        raise ValueError('K must be at least two.')
    a2 = rng.chisquare(K, size=shape)
    b = rng.standard_normal(size=shape)
    c2 = rng.chisquare(K - 1, size=shape)
    return np.stack((a2, np.sqrt(a2)*b, b*b+c2), axis=-1)


def scores(scatter: np.ndarray, weights: np.ndarray, K: int) -> np.ndarray:
    return (scatter[..., 0]*weights[0] + scatter[..., 1]*weights[1]
            + scatter[..., 2]*weights[2]) / (2*K)


def cp_bounds(successes: int, n: int, tail: float) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n-successes+1))
    high = 1.0 if successes == n else float(beta.ppf(1-tail, successes+1, n-successes))
    return low, high


def paired_bounds(n10: int, n01: int, n: int, family: int = 1) -> tuple[float, float]:
    tail = .05 / (4*family)
    lo10, hi10 = cp_bounds(n10, n, tail)
    lo01, hi01 = cp_bounds(n01, n, tail)
    return lo10-hi01, hi10-lo01


def classify(interval: tuple[float, float], margin: float) -> str:
    lo, hi = interval
    if lo >= -margin and hi <= margin:
        return 'equivalent'
    if lo > margin:
        return 'materially_higher'
    if hi < -margin:
        return 'materially_lower'
    return 'unresolved'


def run() -> None:
    if not DESIGN.is_file():
        raise FileNotFoundError(f'Missing experiment design: {DESIGN}')
    if (OUT / 'trials.csv').exists() or (OUT / 'summary.csv').exists():
        raise RuntimeError('Formal outputs already exist; refusing to overwrite.')
    design = json.loads(DESIGN.read_text(encoding='utf-8'))
    model, mc, analysis = design['model'], design['monte_carlo'], design['analysis']
    N, M, K = mc['outer_replicates'], mc['reference_paths_per_replicate'], model['scored_transitions']
    alpha, block = mc['alpha'], mc['block_size']
    cutoff = int(math.floor(alpha*(M+1)))
    n_geometry = len(model['kappa_grid'])*len(model['L_grid'])
    family = n_geometry*(len(design['alternatives'])-1)
    if family != analysis['primary_family_size']:
        raise ValueError('Primary family size mismatch.')
    n_null_rates = 2*len(model['kappa_grid'])*len(model['L_grid'])
    rows = []
    fields = ['geometry_index','kappa','L','alternative','replicate','p_instantaneous','p_finite_step']
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        geom = 0
        for kappa in model['kappa_grid']:
            for L in model['L_grid']:
                H, s1, s2 = model['H'], model['sigma1'], model['sigma2']
                inst = H*np.diag([s1*s1, s2*s2])
                finite = euler_covariance(H,L,kappa,s1,s2)
                warning = max(np.linalg.cond(inst),np.linalg.cond(finite)) > design['coordinates']['condition_number_warning']
                weights_ref = [score_weights(finite,c) for c in (inst,finite)]
                weights_alt = [[score_weights(euler_covariance(H,L,kappa,s1*a['sigma1_multiplier'],s2*a['sigma2_multiplier']),c)
                                for c in (inst,finite)] for a in design['alternatives']]
                rng_ref = np.random.default_rng(np.random.SeedSequence([mc['master_seed'],geom,0]))
                rng_held = np.random.default_rng(np.random.SeedSequence([mc['master_seed'],geom,1]))
                counts = np.zeros((len(design['alternatives']),4),dtype=np.int64)
                max_control_score_gap = 0.0
                max_control_rank_gap = 0.0
                for start in range(0,N,block):
                    count = min(block,N-start)
                    ref = scatter_draws(rng_ref,(count,M),K)
                    held = scatter_draws(rng_held,(count,),K)
                    ref_scores = [scores(ref,w,K) for w in weights_ref]
                    for ai, alt in enumerate(design['alternatives']):
                        held_scores = [scores(held,w,K) for w in weights_alt[ai]]
                        ranks = [1+np.sum(rs >= hs[:,None],axis=1) for rs,hs in zip(ref_scores,held_scores)]
                        reject_inst,reject_finite = [r <= cutoff for r in ranks]
                        counts[ai] += [reject_inst.sum(),reject_finite.sum(),
                                       (reject_finite & ~reject_inst).sum(),(reject_inst & ~reject_finite).sum()]
                        if kappa == 0 or L == 1:
                            gap = max(float(np.max(np.abs(ref_scores[0]-ref_scores[1]))),
                                      float(np.max(np.abs(held_scores[0]-held_scores[1]))))
                            max_control_score_gap = max(max_control_score_gap,gap)
                            max_control_rank_gap = max(max_control_rank_gap,float(np.max(np.abs(ranks[0]-ranks[1]))))
                            if gap > 1e-12 or max_control_rank_gap != 0:
                                raise RuntimeError('Exact covariance-map negative control failed.')
                        writer.writerows({'geometry_index':geom,'kappa':kappa,'L':L,'alternative':alt['name'],
                                          'replicate':start+i,'p_instantaneous':ranks[0][i]/(M+1),
                                          'p_finite_step':ranks[1][i]/(M+1)} for i in range(count))
                for ai,alt in enumerate(design['alternatives']):
                    ninst,nfinite,n10,n01 = map(int,counts[ai])
                    point = paired_bounds(n10,n01,N)
                    simultaneous = paired_bounds(n10,n01,N,family)
                    row = {'geometry_index':geom,'kappa':kappa,'L':L,'eta':abs(kappa)*s1*H/s2,
                           'alternative':alt['name'],'n':N,'reject_instantaneous':ninst,'reject_finite_step':nfinite,
                           'rate_instantaneous':ninst/N,'rate_finite_step':nfinite/N,'n10':n10,'n01':n01,
                           'difference':(nfinite-ninst)/N,'difference_low':point[0],'difference_high':point[1],
                           'simultaneous_low':simultaneous[0],'simultaneous_high':simultaneous[1],
                           'classification':classify(simultaneous,analysis['equivalence_margin']) if ai else 'null_check',
                           'condition_instantaneous':float(np.linalg.cond(inst)),
                           'condition_finite_step':float(np.linalg.cond(finite)),
                           'numerical_warning':bool(warning),
                           'negative_control':bool(kappa == 0 or L == 1),
                           'control_max_score_gap':max_control_score_gap,'control_max_rank_gap':max_control_rank_gap}
                    for name,num in [('instantaneous',ninst),('finite_step',nfinite)]:
                        low,high = cp_bounds(num,N,.025)
                        row[f'rate_{name}_low'],row[f'rate_{name}_high'] = low,high
                        low,high = cp_bounds(num,N,.05/(2*n_null_rates))
                        row[f'null_family_{name}_low'],row[f'null_family_{name}_high'] = (low,high) if ai == 0 else ('','')
                    rows.append(row)
                print(f'Completed geometry {geom+1}/{n_geometry}: kappa={kappa}, L={L}',flush=True)
                geom += 1
    with (OUT / 'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        'geometry_cells': n_geometry,
        'summary_rows': len(rows),
        'trial_rows': n_geometry * len(design['alternatives']) * N,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run'])
    args = parser.parse_args()
    run()


if __name__ == '__main__':
    main()
