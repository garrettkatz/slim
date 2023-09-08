import numpy as np
import matplotlib.pyplot as pt

pts = np.linspace(-3, 3, 30)
A, B = np.meshgrid(pts, pts)

ax = pt.subplot(1,2,1,projection='3d')
ax.plot_surface(A, B, A*B, linewidth=0)
ax.plot_surface(A, B, np.minimum(A**2,B**2)*np.sign(A*B), linewidth=0)

ax = pt.subplot(1,2,2,projection='3d')
ax.plot_surface(A, B, A + B, linewidth=0)
# ax.plot_surface(A, B, np.sign(A+B)*(A+B)**2, linewidth=0)
ax.plot_surface(A, B, 2*np.minimum(A, B), linewidth=0)

# pt.legend()
pt.show()


