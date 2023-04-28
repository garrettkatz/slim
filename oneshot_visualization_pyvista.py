import itertools as it
import numpy as np
import pyvista
import pyvistaqt as pvqt
import matplotlib
import matplotlib.pyplot as pt
import scipy as sp
from scipy.spatial.transform import Rotation

matplotlib.use('qtagg')

back = True
crad = 1

ltms = np.load(f"ltms_3.npz")
Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
X = np.concatenate((X, -X[:,::-1]), axis=1)
X = X.astype(float)
print(X)

K = [7, 5, 6, 4]
flip = [False, True, False, True]

imgs = []
xs = []
y = X[1,:].copy()
for t,k in enumerate(K):
    print(f"{t}...")
    xs.append(X[:,k])
    if flip[t]:
        y[k] *= -1
        y[-k-1] *= -1
    region = (Y == y[:4]).all(axis=1).argmax()
    print(region)
    w = W[region]

    if back:
        p = pvqt.BackgroundPlotter(show=False)
    else:
        p = pyvista.Plotter()
    
    cube = pyvista.Cube(bounds=(-1,+1)*3)
    p.add_mesh(cube, show_edges=True, color=(0,)*3, opacity=1, line_width=20, style='wireframe')

    w_arrow = pyvista.Arrow(start=(0, 0, 0), direction=w, tip_length=0.2, tip_radius=0.1, tip_resolution=50, shaft_radius=0.01, shaft_resolution=50, scale='auto')
    p.add_mesh(w_arrow, color=(.5,)*3)

    print(xs)
    for x in xs:
    
        rot, _ = Rotation.align_vectors(a=x.reshape(1,-1), b=np.array([[0,0,1]]))
        M = np.eye(4)
        M[:3,:3] = rot.as_matrix()
        
        circle = pyvista.Circle(radius=crad, resolution=100).transform(M)
        outline = pyvista.MultipleLines(np.vstack((circle.points, circle.points[:1])))
        arrow = pyvista.Arrow(start=(0, 0, 0), direction=x.flat, tip_length=0.1, tip_radius=0.05, tip_resolution=3, shaft_radius=0.01, shaft_resolution=50, scale='auto')
        
        p.add_mesh(circle, color=(.8,)*3, opacity=1, show_edges=False)
        p.add_mesh(outline, color=(0,)*3, opacity=1, show_edges=True, line_width=20)
        p.add_mesh(arrow, color=(.0,)*3)
    
        # # only seems to show when screenshot scale is 1
        # p.add_point_labels(np.array([[0., 0., 1.]]), ["z"], font_size=50, text_color='black', font_family=None, shadow=False, show_points=False, shape=None, shape_opacity=1, always_visible=True)
    
    # plane intersections
    for x1, x2 in it.combinations(xs, r=2):
        print(x1, x2)
        null = sp.linalg.null_space(np.vstack((x1, x2)))
        print(null)
        # p.add_mesh(pyvista.Line(-null.flatten(), null.flatten()), color=(0,)*3, line_width=20)
        p.add_mesh(pyvista.Cylinder(direction=null.flatten(), radius=.01, height=2*crad), color=(0,)*3)

    p.camera.elevation -= 30
    p.camera.azimuth -= 10

    light = pyvista.Light(position=(3, 2.5, .5), focal_point=(0, 0, 0), color='white')
    # light.positional = True
    # light.cone_angle = 40
    # light.exponent = 10
    # light.intensity = 3
    # light.show_actor()
    p.add_light(light)

    if back:
        # img = p.image
        img = p.screenshot(transparent_background=True, scale=8) # start seeing artifacts at large scale
        imgs.append(img)
    
    else:
    
        p.show()

if back:

    for i, img in enumerate(imgs):
        pt.subplot(1, len(imgs), i+1)
        pt.imshow(img)
        pt.axis('off')

    pt.tight_layout()
    pt.show()
    
