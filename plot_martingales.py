import matplotlib.pyplot as plt
import numpy as np

# Generate non-random data
T = 1000
x = np.arange(0, T)
y1 = np.cumsum(np.random.normal(0.25, size=T)) # Increasing data
y2 = np.cumsum(np.random.normal(-0.25, size=T))
y3 = np.cumsum(np.random.normal(-0.0, size=T))


# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, label='$\hat{Y}_t^0 - \hat{Y}_t^1 > 0$')
ax.plot(x, y2, color='red', label='$\hat{Y}_t^0 - \hat{Y}_t^1 < 0$')
ax.plot(x, y3, color='k', label='$\hat{Y}_t^0 - \hat{Y}_t^1 = 0$')


# Add title and legend
ax.set_title('$\mathcal{K}_t$ over time for different $S_t$')
ax.legend()

# Display the plot
plt.savefig(f'martingales.png', dpi=300, bbox_inches='tight')
