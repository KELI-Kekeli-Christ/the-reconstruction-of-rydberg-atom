# Explication du test Pasqal

## Objectif
J’ai préparé un test pour charger les instances Pasqal présentes dans les dossiers `sqauare`, `freespace` et `triangle`, puis lancer plusieurs formulations sur chaque instance.

## Comment les instances sont chargées

Chaque instance Pasqal est constituée de deux types de fichiers :

- un fichier `.txt` qui contient la matrice `Q` sous forme d’arêtes pondérées
- un fichier `.gml` qui contient les positions initiales des nœuds

### Fichier `.txt`
Le fichier texte commence par une ligne d’en-tête du type :

- `n m`

Puis chaque ligne représente une arête :

- `i j w`

avec `i` et `j` en base 0 dans les fichiers Pasqal.

J’ai ajouté dans `utils.jl` la fonction `read_pasqal_txt_q(txt_path; scale=1000.0)` qui :

- lit `n`
- crée une matrice carrée `Q` de taille `n × n`
- remplit `Q[i,j]` et `Q[j,i]`
- divise les poids par `1000.0` pour obtenir des valeurs réelles compatibles avec tes formulations
- met la diagonale à `0`

### Fichier `.gml`
Les fichiers `.gml` contiennent des positions mais, par demande, le test utilise désormais uniquement les fichiers `.txt`.
J’ai conservé `read_pasqal_gml_positions` si tu veux l’utiliser plus tard, mais le runner `test.jl` ignore désormais les `.gml` et génère toujours un `x_init` aléatoire.

### Chargement complet d’une instance
La fonction `load_pasqal_instance(txt_path; gml_path=nothing, scale=1000.0)` renvoie :

- `Q`
- `x_ref`
- `n`
- `L`
- les chemins des fichiers chargés

La fonction `load_pasqal_folder(folder)` permet aussi de charger automatiquement toutes les instances d’un dossier.

## Adaptations faites au code

### Dans `utils.jl`
J’ai ajouté :

 - parser mis à jour pour ignorer les lignes diagonales ou avec poids non-positifs (ex: `6 6 -581`). Seules les lignes `i j w` avec `i != j` et `w > 0` sont prises en compte.

J’ai aussi modifié :
- `L_optimal_from_Q` pour ne prendre que les poids positifs

### Dans `formulations.jl`
Tes matrices Pasqal sont creuses, donc il fallait éviter de boucler naïvement sur toutes les paires `i < j` pour l’objectif.

J’ai donc adapté les formulations pour travailler sur les arêtes actives seulement via `active_pairs(Q)`.

Concrètement :

- les objectifs n’utilisent plus que les paires où `Q[i,j] > 0`
- les valeurs initiales des variables `r`, `d`, etc. ne sont créées que sur ces arêtes
- `solve_mds` a été adapté pour ne pas planter sur les zéros de `Q`

Pour `solve_mds`, j’ai ajouté une stratégie simple :

- on calcule d’abord les distances cibles seulement sur les arêtes actives
- pour les paires absentes, on remplit avec une distance de secours (`fallback_d`) afin de pouvoir faire le double centrage

## Ce que fait `test.jl`
Le fichier `test.jl` est maintenant un petit runner de validation.

Il :

1. charge les dossiers Pasqal
2. charge chaque instance `.txt` + `.gml`
3. calcule `MDS_Exact`
4. lance plusieurs formulations dessus
5. affiche pour chaque solveur :
   - le statut
   - l’objectif
   - l’erreur Rydberg réelle

## Formulations testées
Dans le script de test, j’ai inclus :

- `solve_new_standard`
- `solve_simple`
- `solve_simple_square`
- `solve_taylor`

Tu peux en ajouter d’autres facilement dans la liste.

## Résultat obtenu
Le test a bien marché sur une instance Pasqal réelle :

- `solve_standard` a donné un résultat
- `solve_new_standard` a convergé
- `solve_simple` a convergé

Donc la chaîne de chargement + adaptation des formulations fonctionne.
