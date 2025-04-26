import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(
    origin_samples,
    target_samples,
    origin_label="origin samples",
    target_label="target samples",
    main_title="sample comparison",
    bins=30,
):
    """
    plots normalized histograms for two sets of samples on separate subplots,
    without legends within each subplot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=false)
    fig.suptitle(main_title, fontsize=14)

    # plot origin histogram
    axs[0].hist(origin_samples, bins=bins, density=true, alpha=0.7, color="salmon")
    axs[0].set_title(origin_label)
    axs[0].set_ylabel("density")
    axs[0].grid(axis="y", linestyle="--", alpha=0.7)

    # plot target histogram
    axs[1].hist(target_samples, bins=bins, density=true, alpha=0.7, color="skyblue")
    axs[1].set_title(target_label)
    axs[1].set_xlabel("value")
    axs[1].set_ylabel("density")
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ---- part A ----

# parameters for the target distribution
a = 2
n = 10000
inverse_uni_cdf = lambda v: a * v

# generate uniform samples from u(0,1)
uni_samples= np.random.uniform(0, 1, n)

# generate samples from the target distribution using inverse transform sampling
target_samples = inverse_uni_cdf(uni_samples)

# plot the histogram of the generated samples
plot_histograms(
    uni_samples,
    target_samples,
    origin_label="origin samples - u(0,1)",
    target_label="target samples - u(0,2)",
    main_title="histograms of 10^4 transformed samples from u(0,1) to u(0,2)",
    bins=30,
)


# ---- part B ----

# parameters for the target distribution
n = 10000
exp_cdf = lambda x: np.where(x >= 0, 1 - np.exp(-x), 0)
exp_cdf_inv = lambda v: -np.log(1 - v)  # inverse cdf for the exponential distribution

# generate uniform samples from u(0,1)
uni_samples = np.random.uniform(0, 1, n)

# generate samples from the exponential distribution
exp_samples = exp_cdf_inv(uni_samples)
# transform the samples to u(0,1) using the cdf of the exponential distribution
uni_samples = exp_cdf(exp_samples)

# plot the histograms of the original and transformed samples
plot_histograms(
    exp_samples,
    uni_samples,
    origin_label="origin samples - exp(1)",
    target_label="target samples - u(0,1)",
    main_title="histograms of 10^4 transformed samples from exponential to u(0,1)",
    bins=30,
)


# ---- Part C ----

# Parameters for the target distribution
N = 10000
GAMMA_CDF = lambda x: np.where(x >= 0, 1 - 4 * np.exp(-4 * x) * (x + 0.25), 0)
BETA_CDF_INV = lambda v: 1 - (1 - v) ** (1 / 5)

# Generate samples from the Gamma(2,4) distribution
gamma_samples = np.random.gamma(shape=2, scale=0.25, size=N)

# Transform the samples to U(0,1) using the CDF of the gamma distribution
uni_samples = GAMMA_CDF(gamma_samples)

# Transform the samples to Beta(1, 5)
beta_samples = BETA_CDF_INV(uni_samples)

# Plot the histograms of the original and transformed samples
plot_histograms(
    gamma_samples,
    beta_samples,
    origin_label="Origin Samples - Gamma(2,4)",
    target_label="Target Samples - Beta(1,5)",
    main_title="Histograms of 10^4 Transformed Samples from Gamma to Beta",
    bins=30,
)