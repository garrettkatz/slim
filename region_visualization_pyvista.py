import itertools as it
import numpy as np
import pyvista
import pyvistaqt as pvqt
import matplotlib
import matplotlib.pyplot as pt
import scipy as sp
from scipy.spatial.transform import Rotation

matplotlib.use('qtagg')
matplotlib.rcParams['text.usetex'] = True

back = True
crad = 1
line_width = 2

window_size= (1024, 720)

ltms = np.load(f"ltms_3.npz")
Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
X = np.concatenate((X, -X[:,::-1]), axis=1)
X = X.astype(float)
print(X)

y = X[1,:].copy()

if back:
    p = pvqt.BackgroundPlotter(window_size=window_size, show=False, lighting='three lights')
else:
    p = pyvista.Plotter(window_size=window_size, lighting='three lights')

# coordinate axes
# for w in W[[2,6]]:
for w in np.eye(3):
    w_arrow = pyvista.Arrow(start=(0, 0, 0), direction=w, tip_length=0.15, tip_radius=0.05, tip_resolution=4, shaft_radius=0.01, shaft_resolution=50, scale=1.5)
    p.add_mesh(w_arrow, show_edges=True, color=(1.,)*3, edge_color='black', line_width=line_width)

for x in X.T:

    opacity = 1

    rot, _ = Rotation.align_vectors(a=x.reshape(1,-1), b=np.array([[0,0,1]]))
    M = np.eye(4)
    M[:3,:3] = rot.as_matrix()
    
    circle = pyvista.Circle(radius=crad, resolution=100).transform(M)
    outline = pyvista.MultipleLines(np.vstack((circle.points, circle.points[:1])))
    # arrow = pyvista.Arrow(start=(0, 0, 0), direction=x.flat, tip_length=0.1, tip_radius=0.05, tip_resolution=3, shaft_radius=0.01, shaft_resolution=50, scale='auto')

    p.add_mesh(circle, color=(.8,)*3, opacity=opacity, show_edges=False)
    # p.add_mesh(circle, color=(.0,)*3, opacity=.9, show_edges=False)
    p.add_mesh(outline, color=(0,)*3, opacity=opacity, show_edges=True, line_width=line_width)

    # # only seems to show when screenshot scale is 1
    # p.add_point_labels(np.array([[0., 0., 1.]]), ["z"], font_size=50, text_color='black', font_family=None, shadow=False, show_points=False, shape=None, shape_opacity=1, always_visible=True)

# plane intersections
int_xs = X.T
for x1, x2 in it.combinations(int_xs, r=2):
    print(x1, x2)
    null = sp.linalg.null_space(np.vstack((x1, x2)))
    print(null)
    if null.shape[1] > 1: continue # x1, x2 colinear
    # p.add_mesh(pyvista.Line(-null.flatten(), null.flatten()), color=(0,)*3, line_width=line_width)
    p.add_mesh(pyvista.Cylinder(direction=null.flatten(), radius=.01, height=2*crad), color=(0,)*3)

# cube outline
cube = pyvista.Cube(bounds=(-1,+1)*3)
p.add_mesh(cube, show_edges=True, color=(0,)*3, opacity=1, line_width=line_width, style='wireframe')


p.camera.elevation += 0#15
p.camera.azimuth -= 15
p.camera.position = tuple(pos*.8 for pos in p.camera.position)

# light = pyvista.Light(position=(3, 2.5, .5), focal_point=(0, 0, 0), color='white')
# light.positional = True
# light.cone_angle = 40
# light.exponent = 10
# light.intensity = 3
# light.show_actor()
# p.add_light(light)

if back:
    # img = p.image
    # img = p.screenshot(transparent_background=True, scale=8) # start seeing artifacts at large scale
    img = p.screenshot(transparent_background=True, scale=2)
    print("img", img.shape)

else:

    p.show()

if back:

    # pt.figure(figsize=(2*6.5,2*1.5))
    pt.figure(figsize=(2*2.5,2*2.5))

    pt.imshow(img)
    pt.axis('off')
    # pt.xticks([], [])
    # pt.yticks([], [])
    # pt.title(f"$t = {i+1}$")

    pt.tight_layout(pad=0)
    pt.savefig("regions.png")
    pt.show()
    


