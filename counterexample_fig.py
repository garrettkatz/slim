import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp 
from matplotlib import rcParams
from check_span_rule import *

# rcParams['font.family'] = 'serif'
# rcParams['text.usetex'] = True

solver = "GUROBI"
N = 8

# load the regions and boundaries
with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# sort ascending by number of boundaries
sorter = np.argsort(B.sum(axis=1))

# extract the counterexample
sample = sorter[[215, 503]] # indices after sorting
Y = Y[sample]
B = B[sample]
W = W[sample]

# double-check infeasibility
result = check_span_rule(X, Y, B, W, solver=solver, verbose=True)
status, u, g, D, E = result
assert status != "optimal"

# extract relevant vertices
X = X[B.any(axis=0)]

# fig params
bx = .4 # size of the sign boxes
rd = .3 # radius of the node circles
fs = 8 # font-size

# setup horizontal image offsets at each node relative to parent
children = {}
offsets = {}
for (n, p, x, y) in E:
    offsets[n] = children.get(p, 0)
    children[p] = offsets[n] + 2*(rd + bx)

# setup image coordinates at each node (assumes parents come before children)
coords = {0: (0, +1)} # root, +1 before first matching x index
for (n, p, x, y) in E:
    k = np.argmax((X == x).all(axis=1))
    c, _ = coords[p]
    coords[n] = (c + offsets[n], -k)

# setup the figure
pt.figure(figsize=(3,5))

# draw the edges
for (n, p, x, y) in E:
    nc, nk = coords[n]
    pc, pk = coords[p]
    if nc == pc:
        pt.plot([pc, nc], [pk, nk], 'k-', zorder=-100)
    else:
        a = mp.Arc((pc, nk), 2*(nc-pc), 2*(pk-nk), angle=0., theta1=0., theta2=90., color='k')
        pt.gca().add_patch(a)

# draw the nodes
for n, (nc, nk) in coords.items():
    pt.gca().add_patch(mp.Circle((nc, nk), rd, ec='k', color='w', fill=True))
    pt.text(nc - .67*rd, nk - .5*rd, str(n), fontsize=fs)

# draw the vectors
pt.text(bx*(N/2 - N - 4) - bx/3, +1, "$x$")
for k, n in it.product(range(X.shape[0]), range(X.shape[1])):
    if X[k,n] == +1:
        s, ec, fc = '+', 'k', 'w'
    else:
        s, ec, fc = '$-$', 'w', 'k'
    x, y = bx*(n - N - 4), -k
    pt.gca().add_patch(mp.Rectangle((x-bx/2, y-bx/2), bx, bx, ec='k', fc=fc, fill=True))
    pt.text(x - .33*bx, y - .33*bx, s, fontsize=8, color=ec)

# draw the labels
for (n, p, x, y) in E:
    nc, nk = coords[n]
    if y == +1:
        s, ec, fc = '+', 'k', 'w'
    else:
        s, ec, fc = '$-$', 'w', 'k'
    x, y = nc - 2*bx, nk
    pt.gca().add_patch(mp.Rectangle((x-bx/2, y-bx/2), bx, bx, ec='k', fc=fc, fill=True))
    pt.text(x - .33*bx, y - .33*bx, s, fontsize=8, color=ec)

pt.axis('equal')
pt.axis('off')
pt.savefig("counterexample_tree.pdf")
pt.show()

