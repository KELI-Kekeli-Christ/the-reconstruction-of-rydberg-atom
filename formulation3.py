import numpy as np
from scipy.optimize import minimize, basinhopping

# --- 1. Paramètres du problème ---
n = 1000      # Nombre de points
dim = 2        # Dimension
C6 = 1      # Constante C6

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

# --- 3. Fonction objectif et son gradient ---
def objective_and_gradient(x_flat, n, dim, Q, C6):
    X = x_flat.reshape((n, dim))
    obj_value = 0.0
    grad_X = np.zeros_like(X)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            norm_sq = np.sum(diff**2)
            
            if norm_sq < 1e-12:
                norm_sq = 1e-12
                
            log_term = np.log(C6 / Q[i, j]) - 3 * np.log(norm_sq)
            
            obj_value += log_term**2
            
            common_factor = -12 * log_term * (diff / norm_sq)
            grad_X[i] += common_factor
            grad_X[j] -= common_factor
            
    return obj_value, grad_X.flatten()

# --- 4. Configuration de l'optimiseur global ---
# On commence n'importe où, très loin de la solution
x0 = np.random.uniform(-50, 50, size=(n * dim))

# On définit le minimiseur local (L-BFGS-B qui utilise notre gradient parfait)
minimizer_kwargs = {
    "method": "L-BFGS-B",
    "jac": True,
    "args": (n, dim, Q_true, C6),
    "options": {"ftol": 1e-15, "gtol": 1e-12}
}

print("Lancement de l'optimisation GLOBALE (Basinhopping)...")

# Basinhopping va sauter de vallée en vallée
res = basinhopping(
    func=objective_and_gradient,
    x0=x0,
    niter=20,                 # Nombre de sauts globaux à tenter
    stepsize=5.0,              # Taille des sauts pour franchir les montagnes
    minimizer_kwargs=minimizer_kwargs
)

# --- 5. Analyse du résultat global ---
print("\n--- ANALYSE DE LA CONVERGENCE GLOBALE ---")
print(f"Statut du solveur : {res.message}")
print(f"Valeur finale de l'objectif : {res.fun:.7e}")

if res.fun < 1e-10:
    print("✅ Succès absolu ! L'algorithme global a trouvé le zéro parfait en sautant les pièges.")
else:
    print("❌ Échec (rare avec cette méthode, augmente 'niter' ou 'stepsize').")

# Reconstruction de la solution trouvée
X_optimal = res.x.reshape((n, dim))
print("\n--- SOLUTION OPTIMALE TROUVÉE ---")
print(X_optimal)