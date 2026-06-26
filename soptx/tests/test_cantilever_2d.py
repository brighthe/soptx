import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

x = np.linspace(-1, 1, 40)
y = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(x, y)

u1 = (np.abs(X) - np.abs(Y)) / (X - Y)
u2 = (X + Y) / (np.abs(X) + np.abs(Y))

# --- 图1：灾难现场（原始公式） ---
fig1 = plt.figure(figsize=(8, 8))
axes0 = fig1.add_subplot(111, projection='3d')
surf1 = axes0.plot_surface(X, Y, u1, cmap='Reds', edgecolor='k', linewidth=0.3, alpha=0.9)
axes0.set_xlabel("x")
axes0.set_ylabel("y")
axes0.set_zlabel("f(x,y)")
axes0.set_zlim(-1.2, 1.2)
plt.tight_layout()
plt.show()

# --- 图2：重构状态（安全公式） ---
fig2 = plt.figure(figsize=(8, 8))
axes1 = fig2.add_subplot(111, projection='3d')
surf2 = axes1.plot_surface(X, Y, u2, cmap='Blues', edgecolor='k', linewidth=0.3, alpha=0.9)
axes1.set_xlabel("x")
axes1.set_ylabel("y")
axes1.set_zlabel("f(x,y)")
axes1.set_zlim(-1.2, 1.2)
plt.tight_layout()
plt.show()