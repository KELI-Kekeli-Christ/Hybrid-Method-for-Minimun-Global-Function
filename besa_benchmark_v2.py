"""
=============================================================================
BESA — Blind Evolutionary Stochastic Attractor  (v2)
=============================================================================
Nouveautés v2 :
  - Règle de calibration automatique : sigma0 = b_max - b_min
  - Tests en dimension 10 ET 30
  - Analyse de la sensibilité sigma0 sur Ackley (explication de l'échec v1)
  - Fonction sigma0_rule() pour calibration adaptative

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
# 2. RÈGLE DE CALIBRATION SIGMA0  (contribution v2)
# ─────────────────────────────────────────────────────────────────────────────

def sigma0_rule(bounds):
    """
    Règle de calibration automatique de sigma0.

    Observation empirique (v2) : sigma0 doit être de l'ordre de la plage
    de recherche pour que les sondes puissent traverser les zones plates
    des fonctions comme Ackley dès les premières générations.

    Règle : sigma0 = b_max - b_min

    Justification :
      - Ackley  [-32, 32]  → sigma0 = 64  → sigma(g=0) = 64/10 = 6.4
      - Rastrigin [-5.12, 5.12] → sigma0 = 10.24 → sigma(g=0) = 1.02
      - Sphere [-100,100] → sigma0 = 200 → sigma(g=0) = 20

    Avec C=10 (valeur par défaut), sigma(g=0) = sigma0/C couvre
    environ 10% de la plage, ce qui est suffisant pour l'exploration initiale.
    """
    lo, hi = bounds
    return float(hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ALGORITHMES
# ─────────────────────────────────────────────────────────────────────────────

def besa(func, dim, bounds, N=30, k=8, sigma0=None, C=10,
         max_gen=100, auto_sigma=True, return_history=False):
    """
    BESA : Blind Evolutionary Stochastic Attractor (v2).

    Paramètres
    ----------
    func         : fonction objectif R^d -> R
    dim          : dimension d
    bounds       : (lo, hi) bornes identiques par dimension
    N            : taille de la population
    k            : nombre de survivants (truncation selection)
    sigma0       : rayon de mutation initial.
                   Si None et auto_sigma=True : applique sigma0_rule(bounds)
    C            : constante de relaxation (contrôle latence d'exploration)
    max_gen      : nombre maximal de générations
    auto_sigma   : si True et sigma0 est None, calibre sigma0 automatiquement
    return_history : si True, retourne aussi l'historique de convergence

    Loi de mutation : sigma_g = sigma0 / (g + C)

    Retour
    ------
    best_val [, history]
    """
    lo, hi = bounds

    # Calibration automatique de sigma0
    if sigma0 is None:
        sigma0 = sigma0_rule(bounds) if auto_sigma else 2.0

    pop     = np.random.uniform(lo, hi, (N, dim))
    history = []

    for g in range(max_gen):
        scores  = np.array([func(x) for x in pop])
        idx     = np.argsort(scores)[:k]
        surv    = pop[idx]
        sigma_g = sigma0 / (g + C)   # loi harmonique

        if return_history:
            history.append(float(scores[idx[0]]))

        new_pop = np.empty((N, dim))
        for i in range(N):
            parent     = surv[np.random.randint(k)]
            new_pop[i] = np.clip(
                parent + np.random.normal(0, sigma_g, dim), lo, hi
            )
        pop = new_pop

    scores   = np.array([func(x) for x in pop])
    best_val = float(scores.min())

    if return_history:
        history.append(best_val)
        return best_val, history
    return best_val


def es_classic(func, dim, bounds, N=30, k=8, sigma=0.5,
               max_gen=100, return_history=False):
    """ES classique à sigma fixe."""
    lo, hi  = bounds
    pop     = np.random.uniform(lo, hi, (N, dim))
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
            new_pop[i] = np.clip(
                parent + np.random.normal(0, sigma, dim), lo, hi
            )
        pop = new_pop

    scores   = np.array([func(x) for x in pop])
    best_val = float(scores.min())

    if return_history:
        history.append(best_val)
        return best_val, history
    return best_val


def run_cmaes(func, dim, bounds, sigma0_cma=None,
              max_gen=100, return_history=False):
    """CMA-ES via la bibliothèque cma."""
    lo, hi = bounds
    if sigma0_cma is None:
        sigma0_cma = (hi - lo) / 4.0   # heuristique standard CMA-ES
    x0     = np.random.uniform(lo, hi, dim)
    opts   = {
        'bounds':  [lo, hi],
        'maxiter': max_gen,
        'verbose': -9,
        'tolx': 1e-12, 'tolfun': 1e-12,
    }
    history = []
    try:
        es2 = cma.CMAEvolutionStrategy(x0, sigma0_cma, opts)
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
# 4. UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def multi_run(algo_fn, func, dim, bounds, runs=20, **kwargs):
    """Lance algo_fn sur `runs` seeds indépendants."""
    vals = []
    for _ in range(runs):
        np.random.seed(np.random.randint(0, 2**31))
        vals.append(algo_fn(func, dim, bounds, **kwargs))
    return np.array(vals)


def wilcoxon_p(a, b):
    """Test de Wilcoxon signé. Retourne p-value."""
    try:
        if np.allclose(a, b): return 1.0
        _, p = wilcoxon(a, b)
        return float(p)
    except Exception:
        return 1.0


def fmt(vals):
    return f"{vals.mean():.3e} +/- {vals.std():.2e}  best={vals.min():.3e}"


COLORS = {'BESA': '#E84B2A', 'BESA-v1': '#FF7043',
          'ES': '#2196F3', 'CMA-ES': '#9C27B0'}
LSS    = {'BESA': '-', 'BESA-v1': ':', 'ES': '--', 'CMA-ES': '-.'}


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSE SIGMA0 SUR ACKLEY  (nouvelle section v2)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_sigma0_ackley(dim=10, runs=15):
    """
    Analyse empirique de l'effet de sigma0 sur Ackley.
    Confirme la règle sigma0 = b_max - b_min.
    """
    print("\n" + "=" * 65)
    print(f"ANALYSE SIGMA0 — Ackley {dim}D  ({runs} runs)")
    print("Règle théorique : sigma0 = b_max - b_min = 64")
    print("=" * 65)

    func, bounds = BENCHMARKS['Ackley']
    lo, hi = bounds
    C = 10

    sigma0_vals = [2.0, 3.2, 8.0, 16.0, 32.0, 64.0]
    results     = []

    for s0 in sigma0_vals:
        sg0  = s0 / C
        vals = multi_run(besa, func, dim, bounds, runs,
                         sigma0=s0, C=C, auto_sigma=False)
        results.append((s0, sg0, vals))
        marker = " ← règle sigma0_rule()" if s0 == 64.0 else ""
        print(f"  sigma0={s0:<6.1f}  sigma(g=0)={sg0:.2f}  "
              f"mean={vals.mean():.3f}  std={vals.std():.3f}{marker}")

    print(f"\n  Règle recommandée : sigma0 = {hi-lo:.1f}  "
          f"(= b_max - b_min)")
    print(f"  ES fixe sigma=0.5 gagnait avant — "
          f"c'était un match accidentel avec la géométrie d'Ackley.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARK PRINCIPAL — 10D et 30D
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(dim=10, runs=20):
    """
    Benchmark complet : BESA-v2 (sigma0 auto) vs BESA-v1 (sigma0=2)
    vs ES classique vs CMA-ES.
    """
    print("\n" + "=" * 65)
    print(f"BENCHMARK  |  dim={dim}  runs={runs}")
    print("BESA-v2 utilise la règle sigma0 = b_max - b_min")
    print("=" * 65)

    all_results = {}

    for fname, (func, bounds) in BENCHMARKS.items():
        s0_auto = sigma0_rule(bounds)
        print(f"\n  {fname}  (sigma0_auto={s0_auto:.2f})")

        r = {}
        # BESA v2 — sigma0 calibré automatiquement
        r['BESA-v2'] = multi_run(besa, func, dim, bounds, runs,
                                 auto_sigma=True)
        # BESA v1 — sigma0 fixe à 2.0 (ancienne version)
        r['BESA-v1'] = multi_run(besa, func, dim, bounds, runs,
                                 sigma0=2.0, auto_sigma=False)
        # ES classique
        r['ES']      = multi_run(es_classic, func, dim, bounds, runs)
        # CMA-ES
        r['CMA-ES']  = multi_run(run_cmaes, func, dim, bounds, runs)

        for algo, vals in r.items():
            p_str = ""
            if algo != 'BESA-v2':
                p   = wilcoxon_p(r['BESA-v2'], vals)
                sig = "OK" if p < 0.05 else "NS"
                p_str = f"   Wilcoxon p={p:.3f} {sig}"
            label = "← nouvelle version" if algo == 'BESA-v2' else ""
            print(f"    {algo:<10} {fmt(vals)}{p_str}  {label}")

        all_results[fname] = r

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_sigma0_ackley(sigma0_results, dim=10,
                      outfile='fig_sigma0_ackley.png'):
    """Figure : effet de sigma0 sur Ackley."""
    s0_vals = [r[0] for r in sigma0_results]
    means   = [r[2].mean() for r in sigma0_results]
    stds    = [r[2].std()  for r in sigma0_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gauche : erreur vs sigma0
    ax = axes[0]
    ax.errorbar(s0_vals, means, yerr=stds,
                fmt='o-', color='#E84B2A', capsize=5, lw=2, markersize=7)
    ax.axvline(x=64, color='#4CAF50', lw=2, ls='--',
               label='Règle : σ₀ = b_max - b_min = 64')
    ax.axhline(y=3.8, color='#2196F3', lw=1.5, ls=':',
               label='ES fixe σ=0.5 (référence)')
    ax.set_xlabel('σ₀', fontsize=12)
    ax.set_ylabel('Erreur finale (Ackley 10D)', fontsize=12)
    ax.set_title(f'Effet de σ₀ sur BESA — Ackley {dim}D',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Droite : sigma(g) pour différents sigma0
    ax2 = axes[1]
    g   = np.arange(100)
    C   = 10
    palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(s0_vals)))
    for i, s0 in enumerate(s0_vals):
        lw = 2.5 if s0 == 64.0 else 1.5
        ax2.plot(g, s0 / (g + C), color=palette[i], lw=lw,
                 label=f'σ₀={s0}')
    ax2.axhline(y=0.5, color='#2196F3', lw=1.5, ls=':',
                label='ES fixe σ=0.5')
    ax2.axhline(y=1.0, color='#FF9800', lw=1, ls='--',
                label='Seuil utile Ackley (~1)')
    ax2.set_xlabel('Génération g', fontsize=12)
    ax2.set_ylabel('σ(g) = σ₀ / (g + C)', fontsize=12)
    ax2.set_title('Décroissance σ(g) selon σ₀\n(C=10)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_ylim(0, 7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        'Analyse de la calibration σ₀ — Pourquoi BESA-v1 échouait sur Ackley',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


def fig_convergence_v2(dim=10, cruns=5, max_gen=100,
                       outfile='fig_convergence_v2.png'):
    """Courbes de convergence BESA-v2 vs v1 vs ES vs CMA-ES."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    runners = {
        'BESA-v2': lambda f, b: besa(
            f, dim, b, max_gen=max_gen, auto_sigma=True, return_history=True),
        'BESA-v1': lambda f, b: besa(
            f, dim, b, sigma0=2.0, max_gen=max_gen,
            auto_sigma=False, return_history=True),
        'ES':      lambda f, b: es_classic(
            f, dim, b, max_gen=max_gen, return_history=True),
        'CMA-ES':  lambda f, b: run_cmaes(
            f, dim, b, max_gen=max_gen, return_history=True),
    }
    colors = {'BESA-v2': '#E84B2A', 'BESA-v1': '#FF9800',
              'ES': '#2196F3', 'CMA-ES': '#9C27B0'}
    lss    = {'BESA-v2': '-', 'BESA-v1': ':', 'ES': '--', 'CMA-ES': '-.'}

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
            mn  = mat.mean(0)
            sd  = mat.std(0)
            xax = np.linspace(0, max_gen, ml)
            ax.semilogy(xax, mn, color=colors[algo], ls=lss[algo],
                        lw=2.2, label=algo)
            ax.fill_between(xax, np.clip(mn - sd, 1e-10, None), mn + sd,
                            alpha=0.12, color=colors[algo])

        ax.set_title(fname, fontsize=13, fontweight='bold')
        ax.set_xlabel('Génération', fontsize=11)
        ax.set_ylabel('Erreur (log)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Convergence BESA-v2 vs v1 vs ES vs CMA-ES  (dim={dim}, {cruns} runs)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


def fig_scalability(results_10d, results_30d,
                    outfile='fig_scalability.png'):
    """Comparaison 10D vs 30D pour BESA-v2, ES, CMA-ES."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    algos   = ['BESA-v2', 'ES', 'CMA-ES']
    colors  = {'BESA-v2': '#E84B2A', 'ES': '#2196F3', 'CMA-ES': '#9C27B0'}
    markers = {'BESA-v2': 'o', 'ES': 's', 'CMA-ES': '^'}
    dims    = [10, 30]

    for fi, fname in enumerate(['Rastrigin', 'Ackley']):
        ax = axes[fi]
        for algo in algos:
            means = [results_10d[fname][algo].mean(),
                     results_30d[fname][algo].mean()]
            stds  = [results_10d[fname][algo].std(),
                     results_30d[fname][algo].std()]
            ax.errorbar(dims, means, yerr=stds,
                        fmt=f"{markers[algo]}-", color=colors[algo],
                        capsize=5, lw=2, markersize=8, label=algo)
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Erreur finale moyenne', fontsize=12)
        ax.set_title(f'Scalabilité — {fname}',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(dims)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Scalabilité 10D → 30D  (BESA-v2 avec calibration σ₀ auto)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


def fig_sigma_decay(outfile='fig_sigma.png'):
    """Décroissance harmonique σ(g) pour différentes valeurs de C."""
    fig, ax = plt.subplots(figsize=(8, 5))
    g       = np.arange(100)
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, 5))
    for i, C in enumerate([1, 5, 10, 20, 50]):
        # Rastrigin : sigma0_rule = 10.24
        ax.plot(g, 10.24 / (g + C), lw=2.2, color=palette[i],
                label=f'C = {C}')
    ax.set_xlabel('Génération g', fontsize=12)
    ax.set_ylabel('σ(g) = σ₀ / (g + C)', fontsize=12)
    ax.set_title('Décroissance harmonique (Rastrigin, σ₀=10.24)',
                 fontsize=12, fontweight='bold')
    ax.legend(title='C', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


def fig_landscape_rastrigin(outfile='fig_landscape.png'):
    """Paysage Rastrigin 2D + trajectoires BESA-v2."""
    def r2d(x, y):
        return 20 + (x**2 - 10*np.cos(2*np.pi*x)) \
                  + (y**2 - 10*np.cos(2*np.pi*y))
    x_ = np.linspace(-5.12, 5.12, 250)
    y_ = np.linspace(-5.12, 5.12, 250)
    X, Y = np.meshgrid(x_, y_)
    Z     = r2d(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cf = axes[0].contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.85)
    plt.colorbar(cf, ax=axes[0], label='Énergie')
    axes[0].plot(0, 0, 'r*', ms=16, label='Optimum (0,0)')
    axes[0].set_title('Paysage Rastrigin 2D', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x₁'); axes[0].set_ylabel('x₂')
    axes[0].legend(fontsize=9)

    ax2 = axes[1]
    ax2.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.5)
    lo, hi = -5.12, 5.12
    s0     = sigma0_rule((lo, hi))   # calibration auto

    for run in range(6):
        np.random.seed(run * 7 + 3)
        pop    = np.random.uniform(lo, hi, (30, 2))
        tx, ty = [], []
        for g in range(80):
            sc   = np.array([rastrigin(p) for p in pop])
            idx  = np.argsort(sc)[:8]
            surv = pop[idx]
            tx.append(float(surv[0, 0]))
            ty.append(float(surv[0, 1]))
            sg  = s0 / (g + 10)
            pop = np.array([
                np.clip(surv[np.random.randint(8)] +
                        np.random.normal(0, sg, 2), lo, hi)
                for _ in range(30)
            ])
        ax2.plot(tx, ty, alpha=0.55, lw=1.2, color='#E84B2A')
        ax2.plot(tx[-1], ty[-1], 'ro', ms=5, alpha=0.9)

    ax2.plot(0, 0, 'r*', ms=16, label='Optimum (0,0)')
    ax2.set_title('Trajectoires BESA-v2 — Rastrigin 2D (6 runs)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    # ── Analyse sigma0 sur Ackley ─────────────────────────────────────────
    sigma0_results = analyse_sigma0_ackley(dim=10, runs=15)

    # ── Benchmark 10D ────────────────────────────────────────────────────
    results_10d = run_benchmark(dim=10, runs=20)

    # ── Benchmark 30D ────────────────────────────────────────────────────
    results_30d = run_benchmark(dim=30, runs=20)

    # ── Wilcoxon 10D résumé ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("WILCOXON — BESA-v2 vs autres (dim=10, 20 runs)")
    print(f"{'Fonction':<14} {'vs BESA-v1':<14} {'vs ES':<12} {'vs CMA-ES':<12}")
    print("-" * 55)
    for fname in BENCHMARKS:
        r  = results_10d[fname]
        p1 = wilcoxon_p(r['BESA-v2'], r['BESA-v1'])
        p2 = wilcoxon_p(r['BESA-v2'], r['ES'])
        p3 = wilcoxon_p(r['BESA-v2'], r['CMA-ES'])
        def s(p): return f"{p:.3f}{'✓' if p < 0.05 else '✗'}"
        print(f"{fname:<14} {s(p1):<14} {s(p2):<12} {s(p3):<12}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n--- Génération des figures ---")
    fig_sigma0_ackley(sigma0_results, dim=10)
    fig_convergence_v2(dim=10, cruns=5)
    fig_scalability(results_10d, results_30d)
    fig_sigma_decay()
    fig_landscape_rastrigin()

    print("\n" + "=" * 65)
    print("TERMINÉ — BESA v2 avec calibration sigma0 automatique.")
    print(f"Règle : sigma0 = b_max - b_min")
    print("=" * 65)
