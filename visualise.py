import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paramètres de simulation pour une paire ---
C6 = 66.0
dist_vraie = 2.5 # On fixe une distance cible parfaite à 2.5 unités

# Calcul des cibles Q_ij pour que le minimum soit exactement à la distance vraie
Q_original = C6 / (dist_vraie**6)
Q_log = Q_original 

# Axe des distances (on regarde ce qu'il se passe de 0.5 à 6 unités de distance)
distances = np.linspace(0.5, 6.0, 500)

# --- 2. Calcul des deux fonctions objectifs et de leurs gradients ---
# Formulation originale : (C6/d^6 - Q)^2
obj_original = (C6 / (distances**6) - Q_original)**2
# Gradient original par rapport à d
grad_original = -12 * (C6 / (distances**6) - Q_original) * (C6 / (distances**7))

# Formulation Log : (log(C6/Q) - 6*log(d))^2
obj_log = (np.log(C6 / Q_log) - 6 * np.log(distances))**2
# Gradient log par rapport à d
grad_log = -12 * (np.log(C6 / Q_log) - 6 * np.log(distances)) * (1.0 / distances)

# --- 3. Génération des graphiques ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ----------------------------------------------------
# Graphique 1 : Formulation Originale (Sans Log)
# ----------------------------------------------------
color_obj = 'red'
ax1.plot(distances, obj_original, color=color_obj, lw=2, label="Objectif (Sans Log)")
ax1.set_xlabel("Distance ||xi - xj||", fontsize=11)
ax1.set_ylabel("Valeur de l'objectif", color=color_obj, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_obj)
ax1.set_ylim(-10, 500)

# Création du deuxième axe y pour le gradient
ax1_grad = ax1.twinx()
color_grad = 'orange'
ax1_grad.plot(distances, grad_original, color=color_grad, lw=1.5, linestyle='--', label="Gradient")
ax1_grad.set_ylabel("Valeur du Gradient", color=color_grad, fontsize=11)
ax1_grad.tick_params(axis='y', labelcolor=color_grad)
ax1_grad.set_ylim(-2000, 2000) # Limité car il explose près de 0

ax1.axvline(x=dist_vraie, color='black', linestyle=':', label=f"Distance cible ({dist_vraie})")
ax1.set_title("Formulation Originale\n(Objectif raide, Gradient instable)", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)

# Regroupement des légendes des deux axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines1_g, labels1_g = ax1_grad.get_legend_handles_labels()
ax1.legend(lines1 + lines1_g, labels1 + labels1_g, loc='upper right')

# ----------------------------------------------------
# Graphique 2 : Formulation Logarithmique
# ----------------------------------------------------
color_obj_log = 'blue'
ax2.plot(distances, obj_log, color=color_obj_log, lw=2, label="Objectif (Avec Log)")
ax2.set_xlabel("Distance ||xi - xj||", fontsize=11)
ax2.set_ylabel("Valeur de l'objectif", color=color_obj_log, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_obj_log)
ax2.set_ylim(-1, 50)

# Création du deuxième axe y pour le gradient log
ax2_grad = ax2.twinx()
color_grad_log = 'purple'
ax2_grad.plot(distances, grad_log, color=color_grad_log, lw=1.5, linestyle='--', label="Gradient")
ax2_grad.set_ylabel("Valeur du Gradient", color=color_grad_log, fontsize=11)
ax2_grad.tick_params(axis='y', labelcolor=color_grad_log)
ax2_grad.set_ylim(-40, 40)

ax2.axvline(x=dist_vraie, color='black', linestyle=':', label=f"Distance cible ({dist_vraie})")
ax2.set_title("Formulation Logarithmique\n(Objectif et Gradient bien lissés)", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)

# Regroupement des légendes
lines2, labels2 = ax2.get_legend_handles_labels()
lines2_g, labels2_g = ax2_grad.get_legend_handles_labels()
ax2.legend(lines2 + lines2_g, labels2 + labels2_g, loc='upper right')

# --- Finalisation de l'affichage ---
plt.suptitle("Comparaison des paysages énergétiques et de leurs gradients", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()