"""
=============================================================================
BESA — Blind Evolutionary Stochastic Attractor
Benchmark complet : BESA vs ES Classique vs CMA-ES
=============================================================================
Dépendances : pip install numpy scipy matplotlib cma
Usage       : python besa_benchmark.py
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import cma
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. FONCTIONS DE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    x = np.asarray(x, dtype=float)
    a = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    b = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return a + b + 20 + np.e

def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere(x):
    return float(np.sum(np.asarray(x, dtype=float)**2))

BENCHMARKS = {
    'Rastrigin':  (rastrigin,  (-5.12,   5.12)),
    'Ackley':     (ackley,     (-32.0,   32.0)),
    'Rosenbrock': (rosenbrock, (-2.048,  2.048)),
    'Sphere':     (sphere,     (-100.0, 100.0)),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. ALGORITHMES
# ─────────────────────────────────────────────────────────────────────────────

def besa(func, dim, bounds, N=30, k=8, sigma0=2.0, C=10,
         max_gen=100, return_history=False):
    """BESA : Blind Evolutionary Stochastic Attractor."""
    lo, hi = bounds
    pop    = np.random.uniform(lo, hi, (N, dim))
    history = []
    for g in range(max_gen):
        scores  = np.array([func(x) for x in pop])
        idx     = np.argsort(scores)[:k]
        surv    = pop[idx]
        sigma_g = sigma0 / (g + C)
        if return_history:
            history.append(float(scores[idx[0]]))
        new_pop = np.empty((N, dim))
        for i in range(N):
            parent     = surv[np.random.randint(k)]
            new_pop[i] = np.clip(parent + np.random.normal(0, sigma_g, dim), lo, hi)
        pop = new_pop
    scores   = np.array([func(x) for x in pop])
    best_val = float(scores.min())
    if return_history:
        history.append(best_val)
        return best_val, history
    return best_val


def es_classic(func, dim, bounds, N=30, k=8, sigma=0.5,
               max_gen=100, return_history=False):
    """ES classique a sigma fixe."""
    lo, hi = bounds
    pop    = np.random.uniform(lo, hi, (N, dim))
    history = []
    for g in range(max_gen):
        scores = np.array([func(x) for x in pop])
        idx    = np.argsort(scores)[:k]
        surv   = pop[idx]
        if return_history:
            history.append(float(scores[idx[0]]))
        new_pop = np.empty((N, dim))
        for i in range(N):
            parent     = surv[np.random.randint(k)]
            new_pop[i] = np.clip(parent + np.random.normal(0, sigma, dim), lo, hi)
        pop = new_pop
    scores   = np.array([func(x) for x in pop])
    best_val = float(scores.min())
    if return_history:
        history.append(best_val)
        return best_val, history
    return best_val


def run_cmaes(func, dim, bounds, sigma0=0.5, max_gen=100, return_history=False):
    """CMA-ES via la bibliotheque cma."""
    lo, hi = bounds
    x0     = np.random.uniform(lo, hi, dim)
    opts   = {'bounds': [lo, hi], 'maxiter': max_gen, 'verbose': -9,
              'tolx': 1e-12, 'tolfun': 1e-12}
    history = []
    try:
        es2 = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es2.stop():
            sols = es2.ask()
            es2.tell(sols, [func(s) for s in sols])
            if return_history:
                history.append(float(es2.result.fbest))
        best_val = float(es2.result.fbest)
    except Exception:
        best_val = float('inf')
    if return_history:
        if not history:
            history = [best_val]
        return best_val, history
    return best_val

# ─────────────────────────────────────────────────────────────────────────────
# 3. UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def multi_run(algo_fn, func, dim, bounds, runs=20, **kwargs):
    vals = []
    for _ in range(runs):
        np.random.seed(np.random.randint(0, 2**31))
        vals.append(algo_fn(func, dim, bounds, **kwargs))
    return np.array(vals)

def wilcoxon_p(a, b):
    try:
        if np.allclose(a, b): return 1.0
        _, p = wilcoxon(a, b)
        return float(p)
    except Exception:
        return 1.0

def summary(vals):
    return f"{vals.mean():.3e} +/- {vals.std():.2e}  (best={vals.min():.3e})"

COLORS = {'BESA': '#E84B2A', 'ES': '#2196F3', 'CMA-ES': '#9C27B0'}
LSS    = {'BESA': '-',        'ES': '--',       'CMA-ES': '-.'}

# ─────────────────────────────────────────────────────────────────────────────
# 4. BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(dim=10, runs=20):
    print("=" * 72)
    print(f"BESA BENCHMARK  |  dim={dim}  runs={runs}")
    print("=" * 72)
    results = {}
    for fname, (func, bounds) in BENCHMARKS.items():
        print(f"\n  {fname}")
        r = {}
        r['BESA']   = multi_run(besa,       func, dim, bounds, runs)
        r['ES']     = multi_run(es_classic, func, dim, bounds, runs)
        r['CMA-ES'] = multi_run(run_cmaes,  func, dim, bounds, runs)
        for algo, vals in r.items():
            p_str = ""
            if algo != 'BESA':
                p = wilcoxon_p(r['BESA'], vals)
                p_str = f"   Wilcoxon p={p:.3f} {'OK' if p < 0.05 else 'NS'}"
            print(f"    {algo:<10} {summary(vals)}{p_str}")
        results[fname] = r
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence(dim=10, cruns=5, max_gen=100, outfile='fig_convergence.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    runners = {
        'BESA':   lambda f,b: besa(f,dim,b,max_gen=max_gen,return_history=True),
        'ES':     lambda f,b: es_classic(f,dim,b,max_gen=max_gen,return_history=True),
        'CMA-ES': lambda f,b: run_cmaes(f,dim,b,max_gen=max_gen,return_history=True),
    }
    for idx, (fname, (func, bounds)) in enumerate(BENCHMARKS.items()):
        ax = axes[idx]
        for algo, runner in runners.items():
            curves = []
            for _ in range(cruns):
                np.random.seed(np.random.randint(0, 2**31))
                _, h = runner(func, bounds)
                curves.append(np.array(h, dtype=float))
            ml  = min(len(c) for c in curves)
            mat = np.clip(np.array([c[:ml] for c in curves]), 1e-10, None)
            mn  = mat.mean(0); sd = mat.std(0)
            xax = np.linspace(0, max_gen, ml)
            ax.semilogy(xax, mn, color=COLORS[algo], ls=LSS[algo], lw=2.2, label=algo)
            ax.fill_between(xax, np.clip(mn-sd, 1e-10, None), mn+sd,
                            alpha=0.15, color=COLORS[algo])
        ax.set_title(fname, fontsize=13, fontweight='bold')
        ax.set_xlabel('Generation'); ax.set_ylabel('Erreur (log)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle(f'Convergence BESA vs ES vs CMA-ES (dim={dim}, {cruns} runs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


def fig_sigma_decay(sigma0=2.0, max_gen=100, outfile='fig_sigma.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    g       = np.arange(max_gen)
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, 5))
    for i, C in enumerate([1, 5, 10, 20, 50]):
        ax.plot(g, sigma0/(g+C), lw=2.2, color=palette[i], label=f'C = {C}')
    ax.set_xlabel('Generation g', fontsize=12)
    ax.set_ylabel('sigma(g)', fontsize=12)
    ax.set_title(f'Decroissance harmonique sigma(g) = sigma0/(g+C)  (sigma0={sigma0})',
                 fontsize=12, fontweight='bold')
    ax.legend(title='C', fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[OK] {outfile}")


def fig_sensitivity_C(func_name='Rastrigin', dim=10, cruns=15,
                      C_values=None, outfile='fig_sensitivity.png'):
    if C_values is None:
        C_values = [1, 2, 5, 10, 20, 50, 100]
    func, bounds = BENCHMARKS[func_name]
    means, stds  = [], []
    print(f"\nSensibilite C sur {func_name} {dim}D :")
    for Cv in C_values:
        vals = multi_run(besa, func, dim, bounds, cruns, C=Cv)
        means.append(vals.mean()); stds.append(vals.std())
        print(f"  C={Cv:<4}  mean={vals.mean():.3f}  std={vals.std():.3f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(C_values, means, yerr=stds, fmt='o-', color='#E84B2A',
                capsize=5, lw=2, markersize=7)
    ax.set_xlabel('C', fontsize=12); ax.set_ylabel('Erreur finale', fontsize=12)
    ax.set_title(f'Sensibilite BESA au parametre C ({func_name} {dim}D)',
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[OK] {outfile}")


def fig_landscape_rastrigin(outfile='fig_landscape.png'):
    def r2d(x, y):
        return 20+(x**2-10*np.cos(2*np.pi*x))+(y**2-10*np.cos(2*np.pi*y))
    x_ = np.linspace(-5.12, 5.12, 250); y_ = np.linspace(-5.12, 5.12, 250)
    X, Y = np.meshgrid(x_, y_); Z = r2d(X, Y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cf = axes[0].contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.85)
    plt.colorbar(cf, ax=axes[0], label='Energie')
    axes[0].plot(0, 0, 'r*', ms=16, label='Optimum (0,0)')
    axes[0].set_title('Paysage Rastrigin 2D', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2'); axes[0].legend()
    ax2 = axes[1]; ax2.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.5)
    for run in range(6):
        np.random.seed(run*7+3); lo, hi = -5.12, 5.12
        pop = np.random.uniform(lo, hi, (30, 2)); tx, ty = [], []
        for g in range(80):
            sc=np.array([rastrigin(p) for p in pop]); idx=np.argsort(sc)[:8]; surv=pop[idx]
            tx.append(float(surv[0,0])); ty.append(float(surv[0,1]))
            sg=2.0/(g+10)
            pop=np.array([np.clip(surv[np.random.randint(8)]+np.random.normal(0,sg,2),lo,hi) for _ in range(30)])
        ax2.plot(tx, ty, alpha=0.55, lw=1.2, color='#E84B2A')
        ax2.plot(tx[-1], ty[-1], 'ro', ms=5, alpha=0.9)
    ax2.plot(0, 0, 'r*', ms=16, label='Optimum (0,0)')
    ax2.set_title('Trajectoires BESA (6 runs)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x1'); ax2.set_ylabel('x2'); ax2.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[OK] {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    run_benchmark(dim=10, runs=20)
    print("\n--- Figures ---")
    fig_convergence(dim=10, cruns=5)
    fig_sigma_decay()
    fig_sensitivity_C()
    fig_landscape_rastrigin()
    print("\nDone.")
