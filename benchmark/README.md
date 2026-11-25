# Benchmark L-variable — Reconstruction géométrique Rydberg

Projet de stage (LMA / programmation Julia) pour **retrouver des positions d’atomes** \(x_i \in \mathbb{R}^2\) à partir d’une matrice d’interactions \(Q\), en optimisant à la fois la géométrie **et** la taille de boîte \(L\).

---

## 1. Problème scientifique

### 1.1 Physique

En physique des atomes de Rydberg, l’interaction entre deux atomes \(i\) et \(j\) s’écrit typiquement :

\[
Q_{ij} \;=\; \frac{C_6}{\|x_i - x_j\|^6}
\]

avec \(C_6 > 0\) (dans le code : `c6 = 1.0` par défaut).

**Inverse** : si \(Q_{ij}\) est connu, la distance cible idéale est

\[
d_{ij} \;=\; \Bigl(\frac{C_6}{Q_{ij}}\Bigr)^{1/6}.
\]

**Distance minimale optimale** (paire la plus forte) :

\[
z_{\min}^{\mathrm{opt}} \;=\; \Bigl(\frac{C_6}{Q_{\max}}\Bigr)^{1/6}.
\]

### 1.2 Optimisation

On cherche \(x \in \mathbb{R}^{n \times 2}\) et \(L > 0\) en résolvant

\[
\min_{x,L}\; f(x) + \alpha\, L
\quad\text{s.c.}\quad
0 \le x_{ik} \le L,
\]

éventuellement avec la contrainte de non-collision

\[
\|x_i - x_j\| \;\ge\; z_{\min}^{\mathrm{opt}}.
\]

- \(\alpha\) contrôle le compromis **fidélité à \(Q\)** vs **boîte petite**.
- \(f\) dépend de la **formulation** (Standard, Simple, Taylor, …).
- Le solveur est **Ipopt** (via JuMP).

### 1.3 Métrique de qualité (indépendante de \(f\))

Après chaque solve, on évalue l’**erreur Rydberg vraie** :

\[
\mathcal{E}_{\mathrm{Ryd}}
\;=\;
\sum_{i<j}
\Bigl(
  \tfrac{C_6}{\|x_i-x_j\|^6} - Q_{ij}
\Bigr)^2.
\]

Important : un status `LOCALLY_SOLVED` avec un **petit `obj`** (objectif NLP) n’implique **pas** un petit `err_rydberg` — surtout pour Taylor.

---

## 2. Formulations

Implémentées dans `test_real/formulations_Lvar.jl` (registre `LVAR_FORMULATIONS`).

| Nom | Objectif \(f(x)\) | Commentaire |
|-----|-----------------|-------------|
| **Standard** | \(\sum (C_6 / z_{ij}^3 - Q_{ij})^2\), \(z=\|x_i-x_j\|^2\) | Résidu Rydberg exact (via \(z^3=r^6\)) |
| **Simple** | \(\sum (r_{ij}-d_{ij})^2\) | Matching de distances |
| **Simple_square** | \(\sum (r_{ij}^2-d_{ij}^2)^2\) | Idem en distances au carré |
| **Taylor** | \(\sum \frac{36}{d_{ij}^{14}}(r_{ij}-d_{ij})^2\) | Approx. quadratique locale de Standard |
| **Taylor_square** | \(\sum \frac{36}{d_{ij}^{14}}(r_{ij}^2-d_{ij}^2)^2\) | Idem, version carrée |

### Poids Taylor

Avec \(C_6=1\) et \(d=Q^{-1/6}\) :

\[
\frac{36}{d^{14}} \;=\; 36\, Q^{7/3}.
\]

Conséquences observées dans les expériences :

1. **\(Q\) grand** (paires fortes) → poids énorme → NLP mal conditionné (`SLOW_PROGRESS`, `ITERATION_LIMIT`).
2. **\(Q\) petit** (paires faibles) → poids ≈ 0 → \(r\) peut s’éloigner de \(d\) (voire collapse sans `zmin`) alors que l’objectif Taylor reste petit, mais \(\mathcal{E}_{\mathrm{Ryd}}\) explose.

Diagnostic dédié : `test_real/diag_taylor_collapse.jl`.

---

## 3. Les trois expériences de données

| Exp | Source \(Q\) | Description |
|-----|--------------|-------------|
| **Exp 2** | **Synthétique exacte** | Générer des positions \(x_{\mathrm{ref}}\), construire \(Q_{ij}=C_6/\|x_i-x_j\|^6\) |
| **Exp 1** | **Aléatoire** | Matrice \(Q\) bruitée / aléatoire (`Q_bruit_dist`) |
| **Exp 3** | **Pasqal** | Instances réelles dans `pasqal/{freespace,sqauare,triangle}/` (`.gml` / `.txt`) |

Pour chaque source, on compare :

| Mode | `use_zmin` | Contrainte | Question |
|------|------------|------------|----------|
| **avec zmin** | `true` | \(\|x_i-x_j\|\ge z_{\min}^{\mathrm{opt}}\) | Quel \(L^\star\) Ipopt trouve-t-il ? |
| **sans zmin** | `false` | boîte seule | \(d_{\min}(x^\star)\) émerge-t-il ≈ \(z_{\min}^{\mathrm{opt}}\) ? |

Critère d’affichage : si \(|d_{\min}-z_{\min}|/z_{\min} <\) `zmin_match_tol` (défaut `0.2`) → `≈zmin ✓`, sinon `≠zmin`.

---

## 4. Structure du dépôt

```
benchmark/
├── README.md                 ← ce fichier
├── utils.jl                  # utilitaires (racine)
├── formulations.jl           # formulations L fixe / MDS
├── deploy_joyeux_Lalpha.sh   # sync + lancement full sur Joyeux
├── deploy_joyeux_xr.sh       # sync + lancement export x/rij
├── pasqal/                   # instances Pasqal
│   ├── freespace/
│   ├── sqauare/
│   └── triangle/
├── article/                  # article / notes
└── test_real/                # ⭐ cœur des expériences L-variable
    ├── config_L_alpha.toml   # ⭐ TOUT configurer ici
    ├── load_config_L_alpha.jl
    ├── formulations_Lvar.jl  # Standard / Simple / Taylor (+ α L)
    ├── utils.jl
    ├── exp_L_alpha_full.jl   # expérience complète (+ scaling)
    ├── exp_L_alpha_xr.jl     # export x* et paires (rij, dij, …)
    ├── exp_L_alpha.jl        # version simple (logs + CSV résumé)
    ├── diag_taylor_collapse.jl
    └── results/              # CSV / logs (local ou récupérés du serveur)
```

---

## 5. Installation

### Prérequis

- Julia ≥ 1.10 (testé avec 1.11 / 1.12)
- Packages : `JuMP`, `Ipopt`, `DataFrames`, `CSV`, `Statistics`, …

Exemple d’installation dans le dossier projet :

```bash
cd /chemin/vers/julia_programming/benchmark
julia -e 'using Pkg; Pkg.activate("."); Pkg.add(["JuMP","Ipopt","DataFrames","CSV"])'
```

Ipopt nécessite une librairie native (souvent installée automatiquement via le package Julia `Ipopt`).

---

## 6. Configuration — `config_L_alpha.toml`

**Fichier central** : [`test_real/config_L_alpha.toml`](test_real/config_L_alpha.toml)

Éditer ce fichier **avant** de lancer. Les scripts le chargent via `load_config_L_alpha.jl`.

### Sections

| Section | Rôle |
|---------|------|
| `[physics]` | `c6`, `dim`, `z_min`, `d_min` |
| `[common]` | instances, `l_init`, `l_chosen`, formulations, tolérance, seed |
| `[full]` | paramètres de `exp_L_alpha_full.jl` |
| `[xr]` | paramètres de `exp_L_alpha_xr.jl` |
| `[simple]` | paramètres de `exp_L_alpha.jl` |

### Exemple minimal

```toml
[full]
n_values = [4, 10]
alphas = [1.0e-6, 1.0e-4, 1.0]
run_exp1 = true    # Q aléatoire
run_exp2 = true    # Q synthétique
run_exp3 = false   # Pasqal
out_dir = "results"
out_csv_name = "exp_L_alpha_full_results_n4_50.csv"
```

### Priorité

```
variables d’environnement  >  config_L_alpha.toml  >  défauts du loader
```

Utile sur le serveur Joyeux : `deploy_joyeux_*.sh` peut encore surcharger `N_VALUES`, `RUN_EXP3`, etc.

Variables d’env reconnues (extrait) :

| Variable | Effet |
|----------|--------|
| `N_VALUES` | ex. `4,10,20` |
| `N_INSTANCES` | nombre de réplicats |
| `ALPHAS` | ex. `1e-6,1e-4,1.0` |
| `FORMS` | ex. `Simple,Taylor` |
| `RUN_EXP1` / `RUN_EXP2` / `RUN_EXP3` | `0` ou `1` |
| `OUT_DIR` | dossier de sortie |
| `PASQAL_ROOT` | racine des instances Pasqal |
| `L_INIT`, `L_CHOSEN`, `ZMIN_MATCH_TOL`, `SEED` | divers |

---

## 7. Lancer en local

Toujours depuis la racine `benchmark/` (ou avec chemins adaptés) :

### 7.1 Expérience complète (recommandée)

```bash
# 1. Éditer test_real/config_L_alpha.toml
# 2. Lancer
julia --project=. test_real/exp_L_alpha_full.jl
```

Ce script :

1. charge la config (`profile=:full`) ;
2. enchaîne Exp3 (Pasqal) / Exp2 (synthétique) / Exp1 (aléatoire) selon les flags ;
3. pour chaque \((\alpha,\) formulation\()\) : **avec zmin** puis **sans zmin** ;
4. affiche `obj` (objectif solveur) **puis** `ry` (erreur Rydberg) ;
5. applique un **scaling** optionnel si \(L^\star > L_{\mathrm{chosen}}\) ;
6. écrit un CSV + un log dans `out_dir`.

### 7.2 Export positions + paires

```bash
julia --project=. test_real/exp_L_alpha_xr.jl
```

Produit :

- `exp_L_alpha_xr_runs.csv` — 1 ligne / solve (coordonnées aplaties) ;
- `exp_L_alpha_xr_pairs.csv` — 1 ligne / paire \((i,j)\) avec `rij`, `dij`, poids Taylor, `collapsed`, etc.

### 7.3 Script simple

```bash
julia --project=. test_real/exp_L_alpha.jl
```

Version plus légère (CSV résumé `exp_L_alpha_results.csv`).

### 7.4 Diagnostic Taylor

```bash
julia --project=. test_real/diag_taylor_collapse.jl
```

Montre pourquoi, sans `zmin`, Taylor peut coller des atomes (`r → 0`) alors que Simple ne le fait pas.

---

## 8. Lancer sur Joyeux (serveur distant)

Scripts prêts :

```bash
# Expérience full (n=4..50, Exp1+Exp2, Pasqal off par défaut dans le script)
bash deploy_joyeux_Lalpha.sh

# Export x / paires (n=4,10) — n’interrompt pas le job full
bash deploy_joyeux_xr.sh
```

Ces scripts :

1. `rsync` le code + `config_L_alpha.toml` + données Pasqal ;
2. stoppent l’ancien PID du job concerné ;
3. relancent en `nohup` ;
4. écrivent logs dans `~/stage_LMA_2026/benchmark_Lalpha/logs/`.

### Suivi

```bash
ssh "$REMOTE_USER@joyeux.univ-avignon.fr" \
  'tail -f $HOME/stage_LMA_2026/benchmark_Lalpha/logs/exp_L_alpha_n4_50_*.out'
```

### Récupérer les résultats

```bash
scp -r "$REMOTE_USER@joyeux.univ-avignon.fr:\
\$HOME/stage_LMA_2026/benchmark_Lalpha/results/" \
  test_real/
```

---

## 9. Fichiers de sortie — colonnes importantes

### 9.1 CSV « runs » (`exp_L_alpha_full_results*.csv`, `*_xr_runs.csv`)

| Colonne | Signification |
|---------|---------------|
| `exp` / `source` | ex. `exp2_Q_exacte_synthetique/avec_zmin` |
| `instance` | réplicat |
| `n` | nombre d’atomes |
| `n_edges` | nombre de paires actives = \(\|\{(i,j): Q_{ij}>0\}\|\) (= \(n(n-1)/2\) si \(Q\) dense) |
| `alpha` | poids \(\alpha\) devant \(L\) |
| `formulation` | Standard / Simple / … |
| `use_zmin` | contrainte zmin on/off |
| `zmin_opt` | \((C_6/Q_{\max})^{1/6}\) |
| `L_found` | \(L^\star\) trouvé |
| `dmin_found` | \(\min_{i<j}\|x_i-x_j\|\) |
| `ratio_dmin_zmin` | `dmin_found / zmin_opt` |
| `err_rydberg` | \(\mathcal{E}_{\mathrm{Ryd}}\) |
| `obj` | valeur de \(f(x)+\alpha L\) au point trouvé |
| `status` | statut Ipopt (`LOCALLY_SOLVED`, `ECHEC`, …) |
| `time_s` | temps wall-clock |

Colonnes **scaling** (full seulement) : `lambda`, `dmin_scaled`, `err_rydberg_scaled`, `scale_note`, …

### 9.2 CSV « pairs » (`exp_L_alpha_xr_pairs.csv`)

Une ligne = une paire d’atomes d’un run.

| Colonne | Signification |
|---------|---------------|
| `i`, `j` | indices |
| `rij` | distance réalisée \(\|x_i-x_j\|\) |
| `dij` | distance cible \((C_6/Q_{ij})^{1/6}\) |
| `abs_r_minus_d` | \(|r_{ij}-d_{ij}|\) |
| `Qij` | interaction |
| `Qhat_C6_over_r6` | \(C_6/r_{ij}^6\) |
| `deltaQ2` | \((Q-\hat Q)^2\) (terme de \(\mathcal{E}_{\mathrm{Ryd}}\)) |
| `taylor_weight_36_d14` | \(36/d^{14}\) |
| `collapsed` | `true` si `rij < 10^{-4}` |
| `xi1,xi2,xj1,xj2` | coordonnées 2D |

### Calcul de `n_edges`

```julia
active_pairs(Q) = [(i,j) for i in 1:n for j in i+1:n if Q[i,j] > 0]
n_edges = length(active_pairs(Q))
```

---

## 10. Interpréter les résultats (guide rapide)

| Observation | Interprétation |
|-------------|----------------|
| `obj` petit, `ry` petit, `ratio≈1` | Bonne reconstruction |
| `obj` petit, `ry` énorme, `ratio≈0` | Solveur « content » sur un mauvais proxy (souvent **Taylor sans zmin**, collapse) |
| `obj` petit, `ry` moyen, `ratio≈1`, `|r-d|` grand sur paires faibles | Taylor ignore les paires à petit \(Q\) (poids \(Q^{7/3}\)) |
| `SLOW_PROGRESS` / `ITERATION_LIMIT` avec paires fortes OK | Mal conditionnement (poids Taylor trop grands) |
| Simple / Simple² sans zmin et `ratio≈1` | Hypothèse validée : \(d_{\min}\) émerge ≈ \(z_{\min}^{\mathrm{opt}}\) |

L’affichage console des scripts montre désormais :

```text
…  obj=1.01e-02  ry=1.45e+109  [ALMOST_LOCALLY_SOLVED]
```

pour comparer directement objectif NLP vs erreur physique.

---

## 11. Scaling post-solve (`exp_L_alpha_full.jl`)

Si \(L^\star > L_{\mathrm{chosen}}\) :

\[
\lambda = \frac{L_{\mathrm{chosen}}}{L^\star},\quad
x'=\lambda x^\star,\quad
z_{\min}'=\lambda z_{\min},\quad
Q' = Q / \lambda^6.
\]

Sinon : pas de scaling (`scale_applied=false`).  
Utile pour comparer des solutions dans une boîte de taille fixée.

---

## 12. Workflow recommandé

1. **Configurer** `test_real/config_L_alpha.toml` (n, α, Exp1/2/3, formulations).
2. **Tester localement** sur petit cas (`n_values = [4]`, 1 instance).
3. **Lancer** `exp_L_alpha_full.jl` ou déployer via `deploy_joyeux_Lalpha.sh`.
4. **Analyser** CSV : colonnes `obj`, `err_rydberg`, `ratio_dmin_zmin`, `status`.
5. Pour le détail géométrique : lancer `exp_L_alpha_xr.jl` et lire `*_pairs.csv`.
6. Si Taylor pose problème : relire §2 / lancer `diag_taylor_collapse.jl`.

---

## 13. Fichiers annexes utiles

| Fichier | Rôle |
|---------|------|
| `rexplain.md` | Notes sur le chargement Pasqal |
| `article/article.pdf` | Article / dérivations (Taylor, etc.) |
| `visualize_solutions.jl` | Visualisation de solutions |
| `benchmark_bruit_aléatoire.jl` | Benchmarks bruit (racine) |

---

## 14. Contact / contexte

Code de benchmark associé au stage LMA 2026 — reconstruction de configurations à partir d’interactions type Rydberg, comparaison de formulations NLP (Standard / Simple / Taylor) avec \(L\) variable et pénalité \(\alpha L\).
