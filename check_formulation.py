import numpy as np
from scipy.optimize import minimize, basinhopping
import sys

# --- 1. Paramètres du problème ---
n = 200  # Nombre de points
dim = 2       # Dimension
C6 = 1.0      # Constante C6

# --- 2. Génération de la "Vérité Terrain" (Ground Truth) ---
np.random.seed(42)
X_true = np.random.uniform(-5, 5, size=(n, dim))

# Calcul de Q_true pour que le minimum global soit exactement à 0
Q_true = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_sq = np.sum((X_true[i] - X_true[j])**2)
            Q_true[i, j] = C6 / (dist_sq**3)

# --- 3. Fonction objectif LOGARITHMIQUE et son gradient ---
def objective_and_gradient_log(x_flat, n, dim, Q, C6):
    X = x_flat.reshape((n, dim))
    obj_value = 0.0
    grad_X = np.zeros_like(X)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            norm_sq = np.sum(diff**2)
            
            if norm_sq < 1e-12:
                norm_sq = 1e-12
                
            log_term = np.log(C6 / Q[i, j]) - 3.0 * np.log(norm_sq)
            obj_value += log_term**2
            
            common_factor = -12.0 * log_term * (diff / norm_sq)
            grad_X[i] += common_factor
            grad_X[j] -= common_factor
            
    return obj_value, grad_X.flatten()

# --- 4. Fonction objectif ORIGINAL (Physique) stabilisée ---
def objective_and_gradient_original(x_flat, n, dim, Q, C6):
    X = x_flat.reshape((n, dim))
    obj_value = 0.0
    grad_X = np.zeros_like(X)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            norm_sq = np.sum(diff**2)
            
            # CORRECTION CRUCIALE : Bloquer la valeur pour éviter l'explosion à l'infini
            if norm_sq < 1e-12:
                norm_sq = 1e-12
                
            inv_dist6 = 1.0 / (norm_sq**3)
            diff_physique = C6 * inv_dist6 - Q[i, j]
            obj_value += diff_physique**2
            
            # STABILISATION NUMÉRIQUE : Un léger epsilon (1e-3) au dénominateur du gradient
            # empêche le solveur de générer des forces infinies près des singularités.
            common_factor = -12.0 * diff_physique * C6 * (inv_dist6 / (norm_sq + 1e-3))
            grad_X[i] += common_factor * diff
            grad_X[j] -= common_factor * diff
            
    return obj_value, grad_X.flatten()

# --- 5. Configuration et exécution de l'optimiseur global (LOG) ---
x0 = np.random.uniform(-50, 50, size=(n * dim))

minimizer_kwargs = {
    "method": "L-BFGS-B",
    "jac": True,
    "args": (n, dim, Q_true, C6),
    "options": {"ftol": 1e-15, "gtol": 1e-12}
}

print("Lancement de l'optimisation GLOBALE (Basinhopping avec formulation LOG)...")
sys.stdout.flush()

res_global = basinhopping(
    func=objective_and_gradient_log,
    x0=x0,
    niter=30,                 
    stepsize=5.0,              
    minimizer_kwargs=minimizer_kwargs
)

# --- 6. Analyse du résultat global intermédiaire ---
print("\n--- ANALYSE DE LA CONVERGENCE GLOBALE (ESPACE LOG) ---")
print(f"Statut du solveur global : {res_global.message}")
print(f"Valeur finale de l'objectif (LOG) : {res_global.fun:.7e}")

# --- 7. Évaluation de contrôle dans la formulation originale ---
print("\n--- INJECTION INTERMÉDIAIRE DANS LA FORMULATION ORIGINALE ---")
score_physique_intermediaire, _ = objective_and_gradient_original(res_global.x, n, dim, Q_true, C6)
print(f"Valeur de l'objectif ORIGINAL avant polissage : {score_physique_intermediaire:.7e}")
sys.stdout.flush()

# --- 8. POLISSAGE LOCAL SUR LA FORMULATION ORIGINALE ---
print("\n--- LANCEMENT DU POLISSAGE LOCAL (Raffinement sur l'espace original) ---")
sys.stdout.flush()

res_polished = minimize(
    fun=objective_and_gradient_original,
    x0=res_global.x, 
    method="L-BFGS-B",
    jac=True,
    args=(n, dim, Q_true, C6),
    options={"maxiter": 150, "ftol": 1e-18, "gtol": 1e-12}  # Gtol élargi pour accepter la convergence
)

# --- 9. Validation finale ---
print("\n--- ANALYSE DE LA CONVERGENCE FINALE ---")
print(f"Statut du polissage : {res_polished.message}")
print(f"Valeur finale de l'objectif ORIGINAL : {res_polished.fun:.7e}")
sys.stdout.flush()

if res_polished.fun < 1e-8:
    print("\n✅ Victoire totale ! L'approche hybride a éliminé le résidu numérique pour n=100.")
else:
    print("\n❌ Échec du raffinement.")