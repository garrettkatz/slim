import numpy as np
import pyvista
import pyvistaqt as pvqt
import matplotlib.pyplot as pt
from scipy.spatial.transform import Rotation

p = pvqt.BackgroundPlotter(show=False)
# p = pyvista.Plotter()

X = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]]).astype(float)

for k,x in enumerate(X):

    rot, _ = Rotation.align_vectors(a=x.reshape(1,-1), b=np.array([[0,0,1]]))
    M = np.eye(4)
    M[:3,:3] = rot.as_matrix()
    
    circle = pyvista.Circle(radius=1, resolution=100).transform(M)
    outline = pyvista.MultipleLines(np.vstack((circle.points, circle.points[:1])))
    arrow = pyvista.Arrow(start=(0, 0, 0), direction=x.flat, tip_length=0.1, tip_radius=0.05, tip_resolution=3, shaft_radius=0.025, shaft_resolution=50, scale=None)
    
    p.add_mesh(circle, color=(.5,)*3, opacity=.5, show_edges=False)
    p.add_mesh(outline, color=(0,)*3, opacity=1, show_edges=True, line_width=20, edge_color='black')
    p.add_mesh(arrow)

    # # only seems to show when screenshot scale is 1
    # p.add_point_labels(np.array([[0., 0., 1.]]), ["z"], font_size=50, text_color='black', font_family=None, shadow=False, show_points=False, shape=None, shape_opacity=1, always_visible=True)
    
# img = p.image
img = p.screenshot(transparent_background=True, scale=10)

# input('.')

pt.imshow(img)
pt.axis('off')
pt.show()

