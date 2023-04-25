import numpy as np
import pyvista
import pyvistaqt as pvqt
import matplotlib.pyplot as pt

mesh = pyvista.Plane().triangulate()
submesh = mesh.subdivide(2, 'linear')
# submesh.plot(show_edges=True)

circle = pyvista.Circle()
# circle.plot(show_edges=False)

p = pvqt.BackgroundPlotter(show=False)
p.add_mesh(submesh)
p.add_mesh(circle)
# p.add_arrows(cent = np.array([[0, 0, 0]]), direction=np.array([[0, 0, 1]]), mag = 1, show_scalar_bar=False)
p.add_mesh(pyvista.Arrow(start=(0, 0, 0), direction=(0, 0, 1), tip_length=0.1, tip_radius=0.05, tip_resolution=50, shaft_radius=0.025, shaft_resolution=50, scale=None))

# try this instead:
# https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_point_labels.html#add-point-labels
# p.add_text("z", position=(0, 0, 1), font_size=18, color=None, font=None, shadow=False, name=None, viewport=False, orientation=0.0, render=True)

img = p.screenshot(transparent_background=True, scale=10)

pt.imshow(img)
pt.show()

# p.show_bounds(grid=True, location='back')
# p.show()

# # Make a grid
# x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
#                       np.linspace(-5, 5, 20),
#                       np.linspace(-5, 5, 5))

# points = np.empty((x.size, 3))
# points[:, 0] = x.ravel('F')
# points[:, 1] = y.ravel('F')
# points[:, 2] = z.ravel('F')

# # Compute a direction for the vector field
# direction = np.sin(points)**3

# # plot using the plotting class
# pl = pyvista.Plotter()
# pl.add_arrows(points, direction, 0.5)
# pl.show()
