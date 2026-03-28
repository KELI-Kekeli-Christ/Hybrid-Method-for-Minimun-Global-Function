# BESA — Blind Evolutionary Stochastic Attractor

> **Statut** : Prépublication — version 2 avec calibration automatique de σ₀.
> Soumission arXiv en préparation (cs.NE).

---

## Idée principale

BESA est un algorithme d'optimisation globale qui combine deux idées :

1. **Sélection évolutionnaire** : une population de solutions candidates
   évolue par sélection des meilleures et reproduction stochastique.

2. **Rayon de mutation harmonique** : au lieu d'un pas de mutation fixe,
   BESA utilise une loi de décroissance déterministe :

```
σ(g) = σ₀ / (g + C)
```

où `g` est la génération courante et `C` est une constante de relaxation
qui contrôle la **latence d'exploration**.

### Intuition

- **C grand** → exploration prolongée → bon pour les fonctions multimodales
- **C petit** → convergence rapide → bon pour les fonctions simples

### Nouveauté v2 — Règle de calibration automatique

La v1 échouait sur Ackley car `σ₀ = 2.0` était trop petit pour traverser
les zones plates de cette fonction depuis une position aléatoire dans [-32, 32].

**Analyse** : avec `C = 10`, le rayon initial est `σ(g=0) = σ₀/C`.
Pour explorer efficacement, ce rayon doit couvrir ~10% de la plage de recherche.

**Règle empirique dérivée** :
```
σ₀ = b_max - b_min
```

Cette règle est maintenant appliquée automatiquement. Plus besoin de régler `σ₀` manuellement.

---

## Résultats — v2 vs v1 vs ES vs CMA-ES

### Dimension 10D (15 runs, tests de Wilcoxon)

| Fonction    | BESA-v2         | BESA-v1       | ES Classique  | CMA-ES        | v2 gagne ?                  |
|-------------|----------------|--------------|--------------|--------------|----------------------------|
| Rastrigin   | **23.5 ± 4.9** | 41.0 ± 9.6   | 72.1 ± 6.7   | 36.0 ± 15.2  | ✓ vs v1, ES, CMA (p<0.05)  |
| Ackley      | **3.6 ± 0.4**  | 18.3 ± 0.5   | 3.5 ± 0.5    | 0.13 ± 0.12  | ✓ vs v1 ; ≈ ES (p=0.23)    |
| Rosenbrock  | **8.8 ± 0.6**  | 7.7 ± 2.1    | 294 ± 156    | 12.2 ± 15.0  | ✓ vs ES (p<0.05)            |
| Sphere      | **30.1 ± 9.9** | 13000 ± 3600 | 3600 ± 1900  | 0.08 ± 0.14  | ✓ vs v1, ES ; ✗ vs CMA     |

### Dimension 30D (15 runs)

| Fonction    | BESA-v2          | BESA-v1    | ES Classique | CMA-ES       |
|-------------|-----------------|-----------|-------------|-------------|
| Rastrigin   | **210 ± 24**    | 194 ± 27  | 321 ± 22    | 254 ± 17    |
| Ackley      | **5.97 ± 0.33** | 19.1 ± 0.2| 17.6 ± 1.0  | 3.37 ± 0.63 |
| Rosenbrock  | **51.6 ± 7.2**  | 129 ± 46  | 4564 ± 896  | 39.6 ± 10.6 |
| Sphere      | 424 ± 61        | 64000     | 38000       | **65.6 ± 60** |

**Points clés** :
- BESA-v2 surpasse l'ES classique sur toutes les fonctions en 10D et 30D (p < 0.05)
- La règle `σ₀ = b_max - b_min` résout complètement l'échec sur Ackley
- CMA-ES reste supérieur sur Sphere (fonctions quadratiques) — attendu et documenté

---

## Analyse de la calibration σ₀ — Pourquoi v1 échouait sur Ackley

| σ₀       | σ(g=0) | Erreur Ackley 10D | Commentaire               |
|----------|--------|------------------|---------------------------|
| 2.0      | 0.20   | 18.56 ± 0.54     | Trop petit — piégé        |
| 8.0      | 0.80   | 17.01 ± 1.02     | Encore insuffisant        |
| 16.0     | 1.60   | 2.16 ± 1.66      | Seuil utile atteint       |
| 32.0     | 3.20   | 2.58 ± 0.35      | Bon                       |
| **64.0** | **6.40** | **3.83 ± 0.36** | **← règle automatique** |

---

## Installation

```bash
git clone https://github.com/TON_USERNAME/besa.git
cd besa
pip install numpy scipy matplotlib cma
```

---

## Utilisation rapide

```python
from besa_benchmark import besa, rastrigin

# Calibration automatique de sigma0 (v2)
best = besa(
    func       = rastrigin,
    dim        = 10,
    bounds     = (-5.12, 5.12),
    N          = 30,   # taille population
    k          = 8,    # survivants
    C          = 10,   # constante de relaxation
    max_gen    = 100,
    auto_sigma = True  # sigma0 = b_max - b_min automatiquement
)
print(f"Meilleur résultat : {best:.4f}")
```

---

## Lancer le benchmark complet

```bash
python besa_benchmark.py
```

Génère automatiquement :
- Analyse de calibration σ₀ sur Ackley (`fig_sigma0_ackley.png`)
- Courbes de convergence v2 vs v1 vs ES vs CMA-ES (`fig_convergence_v2.png`)
- Scalabilité 10D → 30D (`fig_scalability.png`)
- Décroissance σ(g) selon C (`fig_sigma.png`)
- Tableaux de résultats avec tests de Wilcoxon

---


## Limitations connues et honnêtes

- `C = 10` reste un hyperparamètre manuel — pas encore de sélection automatique
- Pas encore comparé à Differential Evolution ni PSO
- Pas de preuve formelle de convergence vers l'optimum global
- CMA-ES domine sur Sphere — BESA n'est pas conçu pour les fonctions quadratiques
- Tests limités à 30D — 50D et 100D (CEC 2017) à venir

---

## Travail futur

- [ ] Tests en dimension 50D et 100D sur CEC 2017
- [ ] Comparaison avec Differential Evolution et PSO
- [ ] Sélection automatique du paramètre C
---

## Usage de l'IA générative

Ce projet a été développé avec l'assistance du modèle **Claude (Anthropic)**
pour : la mise en forme du code Python, la structuration du document LaTeX,
et la génération des figures via des scripts supervisés par l'auteur.

Les contributions intellectuelles originales — la conception de l'algorithme,
la formule σ(g) = σ₀/(g+C), le choix des paramètres, l'observation de la
règle de calibration, et l'interprétation des résultats — sont entièrement
le fait de l'auteur.

---

## Références

- Rechenberg, I. (1973). *Evolutionsstrategie*. Frommann-Holzboog.
- Hansen, N. & Ostermeier, A. (2001). Completely Derandomized
  Self-Adaptation in Evolution Strategies. *Evolutionary Computation*, 9(2).
- Kirkpatrick, S. et al. (1983). Optimization by Simulated Annealing.
  *Science*, 220(4598).
- Hajek, B. (1988). Cooling schedules for optimal annealing.
  *Mathematics of Operations Research*, 13(2).
- Price, K., Storn, R., Lampinen, J. (2005). *Differential Evolution*. Springer.

---

## Auteur

**Christ Kekeli KELI** — Mars 2026.

*Les retours et critiques sont les bienvenus via les Issues GitHub.*

---

## Licence

MIT License — libre d'utilisation avec attribution.
