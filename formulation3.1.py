import numpy as np
from scipy.optimize import minimize, basinhopping

# --- 1. Paramètres du problème ---
n = 10          # Nombre de points
dim = 2        # Dimension
C6 = 66.0      # Constante C6

# --- 2. Génération de la "Vérité Terrain" ---
np.random.seed(42)
X_true = np.random.uniform(-5, 5, size=(n, dim))

# Calcul de Q_true pour la fonction originale (sans log)
Q_true = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_sq = np.sum((X_true[i] - X_true[j])**2)
            Q_true[i, j] = C6 / (dist_sq**3) # ||x_i - x_j||^6

# --- 3. Fonction objectif originale et son gradient ---
def objective_original(x_flat, n, dim, Q, C6):
    X = x_flat.reshape((n, dim))
    obj_value = 0.0
    grad_X = np.zeros_like(X)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            norm_sq = np.sum(diff**2)
            
            if norm_sq < 1e-12:
                norm_sq = 1e-12
                
            # Formule originale sans log
            u_ij = C6 / (norm_sq**3) # C6 / ||x_i - x_j||^6
            diff_term = u_ij - Q[i, j]
            
            # Valeur de l'objectif
            obj_value += diff_term**2
            
            # Gradient analytique original
            # Note l'utilisation de norm_sq**4, qui correspond à la norme puissance 8
            common_factor = -12 * diff_term * (C6 / (norm_sq**4)) * diff
            
            grad_X[i] += common_factor
            grad_X[j] -= common_factor
            
    return obj_value, grad_X.flatten()

# --- 4. Configuration de l'optimiseur global ---
x0 = np.random.uniform(-50, 50, size=(n * dim))

minimizer_kwargs = {
    "method": "L-BFGS-B",
    "jac": True,
    "args": (n, dim, Q_true, C6),
    "options": {"ftol": 1e-15, "gtol": 1e-12}
}

print("Lancement de l'optimisation GLOBALE originale (sans log)...")

res = basinhopping(
    func=objective_original,
    x0=x0,
    niter=30,                  # Légèrement augmenté car sans log l'espace est plus raide
    stepsize=4.0,
    minimizer_kwargs=minimizer_kwargs
)

# --- 5. Analyse du résultat ---
print("\n--- ANALYSE DE LA CONVERGENCE GLOBALE ---")
print(f"Statut du solveur : {res.message}")
print(f"Valeur finale de l'objectif : {res.fun:.7e}")

if res.fun < 1e-10:
    print("✅ Succès absolu ! L'algorithme a convergé vers 0 avec la formulation originale.")
    print(f" {res.x.reshape((n, dim))} ")
else:
    print("❌ Minimum global non atteint. Astuce : Ajuste 'stepsize' ou 'niter'.")