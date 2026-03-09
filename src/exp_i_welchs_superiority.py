import numpy as np
from scipy import stats

def welchs_ttest_from_summary(mean1, std1, n1, mean2, std2, n2):
    """
    Computes a two-tailed Welch's t-test from summary statistics.
    Returns (t_stat, p_value, cohens_d_estimate)
    """
    # Calculate Welch's t-statistic
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    se_diff = np.sqrt(se1**2 + se2**2)
    
    t_stat = (mean1 - mean2) / se_diff
    
    # Degrees of freedom (Welch-Satterthwaite equation)
    df = (se1**2 + se2**2)**2 / ((se1**4 / (n1 - 1)) + (se2**4 / (n2 - 1)))
    
    # Calculate two-tailed p-value
    p_value = 2 * stats.t.sf(np.abs(t_stat), df)
    
    # Calculate Cohen's d (using pooled standard deviation for effect size)
    var_pooled = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    sd_pooled = np.sqrt(var_pooled)
    cohens_d = np.abs(mean1 - mean2) / sd_pooled
    
    return t_stat, p_value, cohens_d, df

if __name__ == "__main__":
    print("=== RESOLUTION 3: Rigorous Statistical Superiority (Welch's t-test) ===")
    
    # 1. Composed Model (Zero-Shot) - 5 Seeds
    mean_comp = 8.31
    std_comp = 3.46
    n_seeds_comp = 5
    
    # 2. Fairly Tuned Baseline Model (100,000 steps) - 3 Seeds from Colab
    colab_100k_seeds = [0.33, 0.29, 0.42]
    mean_base100k = np.mean(colab_100k_seeds)
    std_base100k = np.std(colab_100k_seeds, ddof=1)
    n_seeds_base = 3
    
    # 3. Fairly Tuned HNN Baseline - 1 Seed
    mse_hnn = 573.00
    
    print("\n[TEST 1] Composed (Zero-Shot) vs Fairly Tuned 100k Baseline (Massive Data)")
    
    t_stat, p_value, cohens_d, df = welchs_ttest_from_summary(
        mean_comp, std_comp, n_seeds_comp, 
        mean_base100k, std_base100k, n_seeds_base
    )
    
    print(f"Composed MSE: {mean_comp:.2f} | 100k Monolithic MSE: {mean_base100k:.2f}")
    print(f"Welch's t-statistic: {t_stat:.2f} (df = {df:.2f})")
    print(f"Effect Size (Cohen's d): {cohens_d:.2f}")
    print(f"p-value: {p_value:.10f}")
    
    print("\n=> SCIENTIFIC CONCLUSION:")
    if mean_base100k < mean_comp and p_value < 0.05:
        print("As predicted by the Senior Reviewer, a massively over-parameterized monolithic")
        print("network given 100,000 samples CAN memorize the combined dynamics better than")
        print("our zero-shot composed model.")
        print("Our claim pivots to Data Efficiency: Zero-Shot composition achieves 8.31 error")
        print("with 0 training samples, whereas structural priors (HNN) failed at 573.00.")

