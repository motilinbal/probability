import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(
    samples,
    title="Sample Distribution",
    xlabel="Value",
    ylabel="Density",
    num_bins=20,
    color="skyblue",
    edge_color="black",
    figsize=(10, 5),
):
    """
    Plots a single normalized histogram for a set of samples with distinct borders.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Creates num_bins bins covering the interval [-1, 1]
    bin_edges = np.linspace(-1, 1, num_bins + 1)

    # Plot the histogram with density normalization and edge color
    ax.hist(samples, bins=bin_edges, density=True, alpha=0.7,
            color=color, edgecolor=edge_color) # Added edgecolor
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Set x-axis limits slightly wider to see the end bins clearly
    ax.set_xlim(-1.1, 1.1)

    plt.tight_layout()
    plt.show()


# --- Define the piecewise inverse CDF ---
def inverse_cdf(v):
    """
    Calculates the value of the piecewise inverse CDF:
    F_x(v) =
       -1,        if 0 <= v < 1/4
       4v - 2,    if 1/4 <= v < 3/4
       1,         if 3/4 <= v <= 1
    """
    # Ensure v is a numpy array for vectorized operations
    v = np.asarray(v)

    # Define conditions for each piece
    conditions = [
        (v >= 0) & (v < 0.25),    # 0 <= v < 1/4
        (v >= 0.25) & (v < 0.75), # 1/4 <= v < 3/4
        (v >= 0.75) & (v <= 1)    # 3/4 <= v <= 1
    ]

    # Define the corresponding functions/values for each condition
    functions = [
        lambda x: -1.0,           # Result is -1
        lambda x: 4 * x - 2,      # Result is 4v - 2
        lambda x: 1.0             # Result is 1
    ]

    # Use numpy.piecewise to apply the conditions and functions
    return np.piecewise(v, conditions, functions)

# ---- Main ----

# generate uniform samples from u(0,1)
N = 10000
uni_samples = np.random.uniform(0, 1, N)

# generate samples from the target distribution using inverse transform sampling
target_samples = inverse_cdf(uni_samples)

# plot the histogram of the generated samples
plot_histogram(
    target_samples,
    title="Transformed Samples from U(0,1) To Piecewise Distribution",
)