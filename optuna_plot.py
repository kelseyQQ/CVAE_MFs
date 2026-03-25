# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# ---------------- Load data ----------------
OUT_DIR = "optuna_plot"        

df = pd.read_csv(f"{OUT_DIR}/all_trials.csv")
X = df[['params_latent', 'params_beta']].values
y = df['value'].values

# ---------------- Split into good / bad sets as TPE does ----------------
gamma = 0.1  # Optuna default ~10%
n_good = max(1, int(np.floor(gamma * len(y))))
idx_sorted = np.argsort(-y)  # maximise
good_idx = idx_sorted[:n_good]
bad_idx = idx_sorted[n_good:]

X_good = X[good_idx]
X_bad = X[bad_idx]

# ---------------- Simple 1‑D Gaussian KDE for each dimension -------------
def kde_1d(samples, grid, bw):
    """Return pdf values on grid using simple Gaussian kernels."""
    n = len(samples)
    pdf = np.zeros_like(grid)
    for s in samples:
        pdf += np.exp(-0.5 * ((grid - s) / bw) ** 2)
    pdf /= n * bw * np.sqrt(2 * np.pi)
    return pdf

# Silverman's rule of thumb bandwidth
def bw_silverman(samples):
    s = np.std(samples, ddof=1)
    n = len(samples)
    return 1.06 * s * n ** (-1 / 5) if n > 1 else 1.0

# ---------------- Create a grid ----------------
latent_lin = np.linspace(df['params_latent'].min(), df['params_latent'].max(), 150)
beta_lin = np.linspace(df['params_beta'].min(), df['params_beta'].max(), 150)


# Precompute KDEs for each dimension
bw_lat_good = bw_silverman(X_good[:, 0])
bw_beta_good = bw_silverman(X_good[:, 1])

bw_lat_bad = bw_silverman(X_bad[:, 0])
bw_beta_bad = bw_silverman(X_bad[:, 1])

pdf_lat_good = kde_1d(X_good[:, 0], latent_lin, bw_lat_good)
pdf_beta_good = kde_1d(X_good[:, 1], beta_lin, bw_beta_good)

pdf_lat_bad = kde_1d(X_bad[:, 0], latent_lin, bw_lat_bad)
pdf_beta_bad = kde_1d(X_bad[:, 1], beta_lin, bw_beta_bad)

# Combine to get joint pdf assuming independence (TPE uses product)
l_pdf = np.outer(pdf_beta_good, pdf_lat_good)  # beta rows x latent cols
g_pdf = np.outer(pdf_beta_bad, pdf_lat_bad)

# Avoid divide‑by‑zero
ratio = np.where(g_pdf > 0, l_pdf / g_pdf, 0.0)

# ---------------- Suggested next point ----------------
max_idx = np.unravel_index(np.argmax(ratio), ratio.shape)
suggest_latent = latent_lin[max_idx[1]]
suggest_beta = beta_lin[max_idx[0]]

# ---------------- Plot l/g ratio heatmap ----------------
plt.figure(figsize=(6, 5))
plt.imshow(ratio, origin='lower',
           extent=[latent_lin.min(), latent_lin.max(), beta_lin.min(), beta_lin.max()],
           aspect='auto')
plt.colorbar(label='l(x) / g(x) ratio')
plt.scatter(df['params_latent'], df['params_beta'], c='white', s=10, label='Trials')
# plt.scatter([suggest_latent], [suggest_beta], marker='x', s=120, c='magenta', label='Suggested next')
plt.xlabel('z')
plt.ylabel('beta')
plt.title('TPE l(x) / g(x) Ratio Heatmap')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tpe_ratio_heatmap.png", dpi=150, bbox_inches="tight")

print(f"Suggested next point (approximate TPE): latent={suggest_latent:.3f}, beta={suggest_beta:.3f}")



df = pd.read_csv(f"{OUT_DIR}/all_trials.csv").sort_values("number")

# Identify parameter columns (Optuna exports usually prefixed with 'params_')
param_cols = [col for col in df.columns if col.startswith('params_')]
X = df[param_cols]
y = df['value']

# ---------- 1. Optimization history ----------
plt.figure()
plt.plot(df['number'], df['value'], marker='o')
plt.xlabel('Trial Number')
plt.ylabel('Objective Value')
plt.title('Optimization History')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optimization_history.png", dpi=150, bbox_inches="tight")

# ---------- 2. Best-so-far (Convergence curve, log-y) ----------
best_so_far = df['value'].cummax()      
x = df['number']                        
plt.figure()
plt.plot(x, best_so_far, marker='o')
plt.yscale('log')                       
plt.xlabel('Trial Number')
plt.ylabel('Best Objective Value So Far (log scale)')
plt.title('Convergence Curve (log-y)')
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/convergence_curve_logy.png", dpi=150, bbox_inches="tight")


# ---------- 2. Best-so-far distance to 1 (|1 - convergence|, log-y) ----------
best_so_far = df['value'].cummax()      
x = df['number']

eps = 1e-12                              
dist = np.maximum(np.abs(1.0 - best_so_far), eps)

fig, ax = plt.subplots()
ax.plot(x, dist, '-', linewidth=2)       

ax.set_yscale('log')

finite = dist[dist > eps]
ymin = finite.min() if finite.size else eps
ymax = dist.max()
ax.set_ylim(ymin - 0.005, ymax * 1.25)     
ax.set_xlim(x.min(), x.max()+1)

ax.set_xlabel('Trial Number')
ax.set_ylabel('|1 - best so far| (log scale)')
ax.set_title('Convergence Curve: distance to 1 (log-y)')
ax.grid(True, which='both', alpha=0.3)
# ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/convergence_distance_to1_logy.png",
           dpi=150, bbox_inches="tight")

# ---------- Best-so-far (with y=1 dashed line) ----------
best_so_far = df['value'].cummax()  
x = df['number']

fig, ax = plt.subplots()
ax.plot(x, best_so_far, '-', linewidth=2, label='best so far')

ax.axhline(1.0, linestyle='--', color='red', linewidth=1.5, label='y = 1')
ax.set_yscale('log')

ymin = best_so_far.min()
ymax = max(best_so_far.max(), 1.0)
pad = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlim(x.min(), x.max() + 1)

ax.set_xlabel('Trial Number')
ax.set_ylabel('Best objective value so far')
ax.set_title('Convergence Curve: Best-so-far')
ax.grid(True, which='both', alpha=0.3)
# ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/convergence_best_so_far.png", dpi=150, bbox_inches="tight")


# ---------- 3. Parameter importance ----------
model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X, y)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [param_cols[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Parameter Importance (Random Forest)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/parameter_importance.png", dpi=150, bbox_inches="tight")

# ---------- 4. Slice plots ----------
for col in param_cols:
    plt.figure()
    plt.scatter(df[col], df['value'])
    plt.xlabel(col)
    plt.ylabel('Objective Value')
    plt.title(f'Slice Plot: {col}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/slice_plot_{col}.png", dpi=150, bbox_inches="tight")
