import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = "../monte-carlo/data/vg_prices_all_heavy.csv"

df = pd.read_csv(file_path)

# Create a figure with two subplots side by side
fig = plt.figure(figsize=(20, 8))

# First subplot: 3D surface plot (K vs T)
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_trisurf(df["K"], df["T"], df["price"], cmap="viridis", edgecolor='none', alpha=0.8)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label="Price")
ax1.set_xlabel("K")
ax1.set_ylabel("T")
ax1.set_zlabel("Price")
ax1.set_title("Price Surface: K vs T")

# Second subplot: Histogram of prices
ax2 = fig.add_subplot(122)
ax2.hist(df["price"], bins=100, edgecolor='black', alpha=0.7)
ax2.set_xlabel("Price")
ax2.set_ylabel(f"Number of occurences (total: {len(df)})")
ax2.set_title("Histogram of Prices")

plt.tight_layout()
plt.savefig("visualizations/dataset_visualizations.png")
plt.close()
