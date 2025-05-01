import pandas as pd
import matplotlib.pyplot as plt

file_path = "../monte-carlo/data/vg_prices_all.csv"

df = pd.read_csv(file_path)

# Plotting surfaces of the dataset
## K vs T
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_trisurf(df["K"], df["T"], df["price"], cmap="viridis", edgecolor='none', alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Price")

ax.set_xlabel("K")
ax.set_ylabel("T")
ax.set_zlabel("Price")
ax.set_title("Price Surface: K vs T")

plt.savefig("visualizations/K_vs_T_surface.png")
plt.close()

## Histogram of prices
plt.figure(figsize=(12, 6))
plt.hist(df["price"], bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("Price")
plt.ylabel(f"Number of occurences (total: {len(df)})")
plt.title("Histogram of Prices")
plt.savefig("visualizations/price_histogram.png")
plt.close()
