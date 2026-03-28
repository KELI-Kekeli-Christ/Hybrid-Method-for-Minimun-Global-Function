# BESA — Blind Evolutionary Stochastic Attractor

> **Statut** : Travail en cours — résultats préliminaires expérimentaux.
> Ce projet est une exploration personnelle d'un algorithme d'optimisation évolutionnaire.

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
qui contrôle la **latence d'exploration** — combien de temps l'algorithme
explore largement avant de converger.

### Intuition

- **C grand** → exploration prolongée → bon pour les fonctions multimodales
- **C petit** → convergence rapide → bon pour les fonctions simples

---

## Résultats préliminaires (10D, 20 runs)

| Fonction     | BESA         | ES Classique  | CMA-ES        | BESA gagne ? |
|-------------|-------------|--------------|--------------|-------------|
| Rastrigin   | 42.0 ± 13.2 | 73.7 ± 11.2  | 54.3 ± 23.7  | ✓ vs ES (p=0.000) |
| Rosenbrock  |  7.2 ±  2.1 | 235.0 ± 76.2 |  7.3 ±  1.7  | ✓ vs ES (p=0.000) |
| Ackley      | 18.3 ±  0.5 |   3.8 ±  0.8 | 16.9 ±  7.3  | ✗ |
| Sphere      | 14000 ± 3000| 3644 ± 1400  |   0.1 ±  0.1 | ✗ |

**Honnêteté** :
- BESA bat l'ES classique significativement sur les fonctions multimodales
- CMA-ES reste supérieur sur les fonctions convexes simples
- BESA perd sur Ackley — raison non encore comprise, en cours d'analyse

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

# Minimiser Rastrigin en 10 dimensions
best = besa(
    func    = rastrigin,
    dim     = 10,
    bounds  = (-5.12, 5.12),
    N       = 30,    # taille population
    k       = 8,     # survivants
    sigma0  = 2.0,   # rayon initial
    C       = 10,    # constante de relaxation
    max_gen = 100
)

print(f"Meilleur résultat : {best:.4f}")
```

---

## Lancer le benchmark complet

```bash
python besa_benchmark.py
```

Génère automatiquement :
- Les tableaux de résultats avec tests de Wilcoxon
- Les courbes de convergence
- L'analyse de sensibilité au paramètre C
- La visualisation du paysage Rastrigin 2D

---

## Limitations connues

- Testé uniquement en dimension 10 pour l'instant
- Pas encore comparé à Differential Evolution ni PSO
- Le choix de C est manuel — pas de sélection automatique
- Pas de preuve formelle de convergence
- Perd sur Ackley et Sphere — raisons en cours d'analyse

---

## Travail futur

- [ ] Tests en dimension 30 et 50
- [ ] Comparaison avec Differential Evolution (DE) et PSO
- [ ] Analyse du comportement sur Ackley
- [ ] Sélection automatique de C
- [ ] Soumission sur arXiv (quand le travail sera plus complet)

---

## Références

- Rechenberg, I. (1973). *Evolutionsstrategie*. Frommann-Holzboog.
- Hansen, N. & Ostermeier, A. (2001). Completely Derandomized
  Self-Adaptation in Evolution Strategies. *Evolutionary Computation*, 9(2).
- Kirkpatrick, S. et al. (1983). Optimization by Simulated Annealing.
  *Science*, 220(4598).
- Hajek, B. (1988). Cooling schedules for optimal annealing.
  *Mathematics of Operations Research*, 13(2).

---

## Auteur

**Christ Kekeli KELI** — Projet personnel, février 2026.

*Ce travail est expérimental et en cours de développement.
Les retours et critiques sont les bienvenus via les Issues GitHub.*

---

## Licence

MIT License — libre d'utilisation avec attribution.
