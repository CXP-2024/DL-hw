import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def p_target(x):
    """Target distribution: 1/3 * N(-3,1) + 2/3 * N(3,1)"""
    return (1.0/3.0) * norm.pdf(x, -3, 1) + (2.0/3.0) * norm.pdf(x, 3, 1)

def log_p_target(x):
    return np.log(p_target(x) + 1e-300)

# Inclusive KL (forward KL): KL(p||q) = integral p(x) log(p(x)/q(x)) dx
# Mode-covering: q is forced to cover all modes of p
def inclusive_kl(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    x = np.linspace(-15, 15, 50000)
    dx = x[1] - x[0]
    px = p_target(x)
    log_qx = norm.logpdf(x, mu, sigma)
    integrand = px * (log_p_target(x) - log_qx)
    return np.sum(integrand) * dx

# Exclusive KL (reverse KL): KL(q||p) = integral q(x) log(q(x)/p(x)) dx
# Mode-seeking: q concentrates on a single mode
def exclusive_kl(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    x = np.linspace(-15, 15, 50000)
    dx = x[1] - x[0]
    qx = norm.pdf(x, mu, sigma)
    log_qx = norm.logpdf(x, mu, sigma)
    integrand = qx * (log_qx - log_p_target(x))
    return np.sum(integrand) * dx

# Optimize inclusive KL: KL(p||q) - mode-covering
res_incl = minimize(inclusive_kl, [0.0, 0.0], method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
mu_incl = res_incl.x[0]
sigma_incl = np.exp(res_incl.x[1])

# Optimize exclusive KL: KL(q||p) - mode-seeking
res_excl = minimize(exclusive_kl, [0.0, 0.5], method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
mu_excl = res_excl.x[0]
sigma_excl = np.exp(res_excl.x[1])

print(f"Inclusive KL  (KL(p||q), mode-covering): mu={mu_incl:.4f}, sigma={sigma_incl:.4f}")
print(f"Exclusive KL  (KL(q||p), mode-seeking):  mu={mu_excl:.4f}, sigma={sigma_excl:.4f}")

# Plot
x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, p_target(x), 'k-', linewidth=2.5,
         label=r'Target $p(x) = \frac{1}{3}\mathcal{N}(-3,1) + \frac{2}{3}\mathcal{N}(3,1)$')
plt.plot(x, norm.pdf(x, mu_incl, sigma_incl), 'r-.', linewidth=2,
         label=rf'Inclusive KL$(p\|q)$: $\mu$={mu_incl:.2f}, $\sigma$={sigma_incl:.2f} (mode-covering)')
plt.plot(x, norm.pdf(x, mu_excl, sigma_excl), 'b--', linewidth=2,
         label=rf'Exclusive KL$(q\|p)$: $\mu$={mu_excl:.2f}, $\sigma$={sigma_excl:.2f} (mode-seeking)')

plt.xlabel('x', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Inclusive vs Exclusive KL Divergence Optimization', fontsize=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/efm-vepfs/group-jt/intern/pcx/DL/DL-hw/Assignment4/kl_divergence.pdf', dpi=150)
plt.savefig('/efm-vepfs/group-jt/intern/pcx/DL/DL-hw/Assignment4/kl_divergence.png', dpi=150)
print("Figure saved.")
