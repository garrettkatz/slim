import itertools as it
import numpy as np
import pyvista
import pyvistaqt as pvqt
import matplotlib
import matplotlib.pyplot as pt
import scipy as sp
from scipy.spatial.transform import Rotation
from adjacent_ltms import adjacency

matplotlib.use('qtagg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

back = True
crad = 1
line_width = 2

window_size= (1024, 720)
# window_size= (512, 360)

ltms = np.load(f"ltms_3.npz")
Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
A, K = adjacency(Y, sym=True)

print(W.round().astype(int))

# add noise to W
W_opt = W.copy()
for i in range(W.shape[0]):
    combo = np.random.rand(W.shape[0]) * .1
    W[i] += combo @ W_opt

assert (np.sign(W @ X) == Y).all()

X = np.concatenate((X, -X[:,::-1]), axis=1)
X = X.astype(float)
print(X)

def draw_circles(p, do_outline=True):

    for x in X.T:
    
        opacity = 1
    
        rot, _ = Rotation.align_vectors(a=x.reshape(1,-1), b=np.array([[0,0,1]]))
        M = np.eye(4)
        M[:3,:3] = rot.as_matrix()
        
        circle = pyvista.Circle(radius=crad, resolution=100).transform(M)
        p.add_mesh(circle, color=(.8,)*3, opacity=opacity, show_edges=False)

        if do_outline:
            outline = pyvista.MultipleLines(np.vstack((circle.points, circle.points[:1])))    
            p.add_mesh(outline, color=(0,)*3, opacity=opacity, show_edges=True, line_width=line_width)
    
        # plane intersections
        int_xs = X.T
        for x1, x2 in it.combinations(int_xs, r=2):
            null = sp.linalg.null_space(np.vstack((x1, x2)))
            if null.shape[1] > 1: continue # x1, x2 colinear
            p.add_mesh(pyvista.Cylinder(direction=null.flatten(), radius=.01, height=2*crad), color=(0,)*3)

def draw_weights(p, Ws, opacity=1):

    # weight vectors
    for w in Ws:
        w_arrow = pyvista.Arrow(start=(0, 0, 0), direction=w, tip_length=0.25, tip_radius=0.15, tip_resolution=4, shaft_radius=0.01, shaft_resolution=50, scale=1.1)
        p.add_mesh(w_arrow, show_edges=True, color=(1.,)*3, edge_color='black', line_width=line_width, opacity=opacity)

def set_view(p, scale):
    p.camera.elevation += 0#15
    p.camera.azimuth -= 15
    # p.camera.position = tuple(pos*.925 for pos in p.camera.position)
    p.camera.position = tuple(pos*scale for pos in p.camera.position)

def get_plotter():
    if back:
        p = pvqt.BackgroundPlotter(window_size=window_size, show=False, lighting='three lights')
    else:
        p = pyvista.Plotter(window_size=window_size, lighting='three lights')
    return p

W = W / np.linalg.norm(W, axis=1, keepdims=True)

# A (4.2): solve linprogs for weights
p = get_plotter()
draw_circles(p)
draw_weights(p, W)
set_view(p, .6)
img1 = p.screenshot(transparent_background=True, scale=2)

# B (4.3): identify adjacencies (dashed lines between them)
p = get_plotter()
draw_circles(p, do_outline=True)
# draw_weights(p, W)
# for i in range(len(A)):
#     wi = 1.1*W[i]
#     for j in A[i]:
#         wj = 1.1*W[j]
#         center = (wi + wj)/2
#         direction = wi - wj
#         height = np.linalg.norm(wi - wj)
#         p.add_mesh(pyvista.Cylinder(center, direction, radius=.05, height=height), color=(.5,)*3)
for i in range(len(A)):
    wi = W_opt[i] / np.linalg.norm(W_opt[i]) * .8
    vert = pyvista.Sphere(.05, wi)
    p.add_mesh(vert, color=(.33,)*3, line_width=line_width)    

    for j in A[i]:
        wj = W_opt[j] / np.linalg.norm(W_opt[j]) * .8
        # arrow = pyvista.Arrow(start=wi + .2*(wj-wi), direction=.7*(wj-wi), tip_length=0.1, tip_radius=0.05, tip_resolution=20, shaft_radius=0.025, shaft_resolution=50, scale='auto')
        # p.add_mesh(arrow, show_edges=False, color=(.25,)*3, edge_color='black', line_width=line_width)
        p.add_mesh(pyvista.Cylinder((wi+wj)/2, (wi-wj), radius=.025, height=np.linalg.norm(wi-wj)), color=(.33,)*3)

set_view(p, .75)
img2 = p.screenshot(transparent_background=True, scale=2)


# D (4.5-4.6): optimize span loss (grayscale w converging to solution)
p = get_plotter()
draw_circles(p)
draw_weights(p, W[[7,8]], opacity=.5)
draw_weights(p, W_opt[[7,8]])
set_view(p, .8)
img3 = p.screenshot(transparent_background=True, scale=2)

# C (4.4): limit to canonicals (bold outlines, dashed arrows with S matrix maps)
p = get_plotter()
draw_circles(p)
# draw_weights(p, W, opacity=.5)
draw_weights(p, W_opt[[7,8]])
draw_weights(p, W_opt[[2,0]], opacity=.5)
p.add_mesh(pyvista.Cylinder(direction=(1,0,1), radius=.1, height=2.25), color=(.25,)*3)
set_view(p, .7)
img4 = p.screenshot(transparent_background=True, scale=2)

pt.figure(figsize=(12, 3))

pt.subplot(1,4,1)
pt.imshow(img1)
pt.axis('off')
# pt.title("Section 4.2", fontsize=18)
# pt.text(50, 50, "Regions", fontsize=18)

pt.subplot(1,4,2)
pt.imshow(img2)
pt.axis('off')
# pt.title("Section 4.3", fontsize=18)
# pt.text(50, 50, "Adjacencies", fontsize=18)

pt.subplot(1,4,3)
pt.imshow(img3)
pt.axis('off')
# pt.title("Section 4.5-4.6", fontsize=18)
# pt.text(50, 50, "Optimization", fontsize=18)

pt.subplot(1,4,4)
pt.imshow(img4)
pt.axis('off')
# pt.title("Section 4.4", fontsize=18)
# pt.text(50, 50, "Symmetries", fontsize=18)

pt.tight_layout(pad=0)
pt.savefig("methods.png")
pt.show()


