import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

# Define the parameters for the Cauchy distribution
x0 = 0  # Location parameter
gamma = 1  # Scale parameter

# Generate x values
# The Cauchy distribution has heavy tails, so we choose a reasonable range.
x = np.linspace(-10, 10, 500)

# Calculate the PDF using scipy.stats.cauchy
pdf = cauchy.pdf(x, loc=x0, scale=gamma)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, pdf, label=f'Cauchy (x₀={x0}, γ={gamma})')

# Add labels and title
plt.xlabel('x')
plt.ylabel('Probability Density Function (PDF)')
plt.title('Cauchy Distribution')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

