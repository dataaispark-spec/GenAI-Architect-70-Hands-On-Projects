# 🧪 Lab Exercise: Importance Sampling & Bayesian Inference  

📚 Phase 1 · Course 5 · 📈 Introduction to Generative Models 

🎯 **Goal:** Learn to approximate expectations under complex distributions by sampling from simpler ones—and apply it to rare-event modeling and Bayesian parameter estimation.

---

## 🧠 Objectives

- Grasp the theory of importance sampling  
- Sample from a proposal distribution and compute importance weights  
- Visualize and compare target vs. proposal distributions  
- Estimate expectations and validate against ground truth  
- Measure estimator quality via Effective Sample Size (ESS)  
- Implement self-normalized importance sampling  
- Apply these techniques to rare-event modeling  
- Perform Bayesian inference on a coin-flip model using importance sampling  

---

## 🛠️ Prerequisites

- Python 3.7+  
- Installed libraries: NumPy, Matplotlib, SciPy, Faker (optional for extension)  
- Understanding of basic probability, PDF/CDF, and Monte Carlo methods  
- Jupyter Notebook or Google Colab environment  

---

## 🔧 Step 1: Setup Environment

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
np.random.seed(42)
```

---

## 📐 Step 2: Define & Visualize Distributions

### 2.1 Simple Example  
- **Target** p(x): Normal(0,1)  
- **Proposal** q(x): Normal(2,1)  
- **Function** f(x): x²  

```python
def p(x): return norm.pdf(x, 0, 1)
def q(x): return norm.pdf(x, 2, 1)
f = lambda x: x**2

xs = np.linspace(-4, 6, 500)
plt.plot(xs, p(xs), label='Target p(x)', color='red')
plt.plot(xs, q(xs), label='Proposal q(x)', color='blue')
plt.title('Target vs. Proposal')
plt.legend(); plt.show()
```

---

## 🎲 Step 3: Standard Importance Sampling

1. Draw _N_ samples from q(x)  
2. Compute weights wᵢ = p(xᵢ) / q(xᵢ)  
3. Estimate Eₚ[f(x)] ≈ (1/N) ∑ f(xᵢ)·wᵢ  

```python
N = 10_000
samples = np.random.normal(2, 1, N)
weights = p(samples) / q(samples)
est_is = np.mean(f(samples) * weights)
print("Importance Sampling Estimate:", est_is)
```

---

## 📈 Step 4: Compare with Monte Carlo via Rejection Sampling

Implement rejection sampling from p(x) and compare estimates:

```python
accepted = []
max_ratio = max(p(xs)/q(xs))
while len(accepted) < N:
    x = np.random.normal(2, 1)
    if np.random.rand() < (p(x) / (q(x) * max_ratio)):
        accepted.append(x)
mc_est = np.mean(np.array(accepted)**2)
print("Rejection Sampling Estimate:", mc_est)
```

Run both estimators multiple times and compare their variance.

---

## 📏 Step 5: Effective Sample Size (ESS)

Compute ESS to measure weight concentration:

\[
\mathrm{ESS} = \frac{\bigl(\sum w_i\bigr)^2}{\sum w_i^2}
\]

```python
ess = (weights.sum()**2) / np.sum(weights**2)
print("Effective Sample Size:", ess)
```

---

## 🔄 Step 6: Self-Normalized Importance Sampling

Normalize weights and recompute:

```python
w_norm = weights / weights.sum()
est_snis = np.sum(f(samples) * w_norm)
print("Self-Normalized IS Estimate:", est_snis)
```

Compare standard vs self-normalized estimates.

---

## 🌟 Step 7: Rare Event Modeling

Model a skewed target for rare events:

```python
def rare_target(x):
    return 0.9 * norm.pdf(x, 0, 1) + 0.1 * norm.pdf(x, 5, 0.5)
def rare_prop(x): return norm.pdf(x, 2, 2)

samples_r = np.random.normal(2, 2, N)
w_r = rare_target(samples_r) / rare_prop(samples_r)
rare_est = np.mean(f(samples_r) * w_r)
print("Rare Event E[x²]:", rare_est)
```

Visualize the skew and discuss why importance sampling helps.

---

## 🧠 Step 8: Bayesian Inference – Coin Bias

Infer posterior of θ given 7 heads, 3 tails.  
- Prior: Beta(2,2)  
- Likelihood: θ⁷(1−θ)³  
- Proposal: Beta(5,5)

```python
def unnorm_post(theta):
    return beta.pdf(theta, 2, 2) * (theta**7 * (1-theta)**3)
def prop_beta(theta): return beta.pdf(theta, 5, 5)

M = 5_000
thetas = np.random.beta(5, 5, M)
w = unnorm_post(thetas) / prop_beta(thetas)
w_norm = w / w.sum()
theta_mean = np.sum(thetas * w_norm)
print("Posterior Mean Estimate:", theta_mean)

# Plot true vs. reweighted
grid = np.linspace(0,1,200)
true_post = beta.pdf(grid, 2+7, 2+3)
plt.plot(grid, true_post, label='True Posterior')
plt.hist(thetas, bins=50, density=True, alpha=0.5, label='Proposal')
plt.plot(grid, (unnorm_post(grid)/prop_beta(grid)), label='Reweighted (unnorm.)')
plt.legend(); plt.title('Importance Sampling for Coin Bias'); plt.show()
```

---

## ❓ Reflection Questions

1. How does ESS reflect proposal–target mismatch?  
2. Why might self-normalized IS be preferred in practice?  
3. What if q(x) fails to cover p(x)’s support?  
4. How would you choose or adapt q(x) for better performance?  
5. In what GenAI scenarios (e.g., RAG, VAE) can importance sampling be applied?

---

## 📚 Extended Resources

- Bishop, C. “Pattern Recognition and Machine Learning”  
- Chen et al. “Monte Carlo Methods in Bayesian Computation”  
- PyMC3 & TensorFlow Probability tutorials on importance sampling  
- SDV (Synthetic Data Vault) for structured data experiments  

---

## ✅ Completion Checklist

- [ ] Visualized p(x) and q(x)  
- [ ] Implemented standard & self-normalized IS  
- [ ] Compared with rejection sampling  
- [ ] Computed ESS  
- [ ] Modeled a rare event distribution  
- [ ] Performed Bayesian inference via IS  
- [ ] Answered all reflection questions  

---

> Congratulations! You’ve unlocked a powerful tool for approximating complex distributions—laying the groundwork for advanced generative modeling and probabilistic AI systems.
