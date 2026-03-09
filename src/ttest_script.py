import scipy.stats as stats
import numpy as np

# Composed model stats (Neuarl)
mean1 = 8.3
std1 = 3.5
n1 = 5

# Baseline 10k stats
mean2 = 7.7
std2 = 2.2
n2 = 5

# Welch's t-test from summary statistics
t_stat, p_val = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

if p_val > 0.05:
    print("Result: NOT significantly different (p > 0.05).")
else:
    print("Result: Significantly different (p < 0.05).")
