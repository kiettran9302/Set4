import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Setup parameters
n_values = np.arange(1, 101) # n ranges from 1 to 100
tau = 0.05

# 2. Calculate metrics based on derivations
# Mean of difference (Bias) = -2/n
bias = -2 / n_values

# Variance of difference = 1/n^2
dev_variance = 1 / (n_values ** 2)

# RMS = sqrt(Bias^2 + Variance) = sqrt(4/n^2 + 1/n^2) = sqrt(5)/n
rms = np.sqrt(5) / n_values

# 3. Calculate Tail Probabilities
# The deviation D is Normal with mean = bias, std_dev = sqrt(variance) = 1/n
mean_diff = bias
std_diff = np.sqrt(dev_variance)

# P(D < -tau)
prob_lower = norm.cdf(-tau, loc=mean_diff, scale=std_diff)

# P(D > tau) = 1 - P(D <= tau)
prob_upper = 1 - norm.cdf(tau, loc=mean_diff, scale=std_diff)

# 4. Plotting
plt.figure(figsize=(12, 10))

# Plot Bias
plt.subplot(2, 2, 1)
plt.plot(n_values, bias, label='Bias')
plt.title('Bias vs Sample Size (n)')
plt.xlabel('n')
plt.ylabel('Bias')
plt.grid(True)

# Plot Deviation Variance
plt.subplot(2, 2, 2)
plt.plot(n_values, dev_variance, label='Deviation Variance', color='orange')
plt.title('Deviation Variance vs Sample Size (n)')
plt.xlabel('n')
plt.ylabel('Variance')
plt.grid(True)

# Plot RMS
plt.subplot(2, 2, 3)
plt.plot(n_values, rms, label='RMS', color='green')
plt.title('RMS vs Sample Size (n)')
plt.xlabel('n')
plt.ylabel('RMS')
plt.grid(True)

# Plot Tail Probabilities
plt.subplot(2, 2, 4)
plt.plot(n_values, prob_lower, label=r'$P(D < -\tau)$')
plt.plot(n_values, prob_upper, label=r'$P(D > \tau)$', linestyle='--')
plt.title(f'Tail Probabilities (tau={tau})')
plt.xlabel('n')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()