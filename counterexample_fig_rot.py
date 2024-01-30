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
bx = .6 # size of the sign boxes
rd = .35 # radius of the node circles
fs = 18 # font-size

# setup horizontal image offsets at each node relative to parent
children = {}
offsets = {}
for (n, p, x, y) in E:
    offsets[n] = children.get(p, 0)
    children[p] = offsets[n] + rd + 2*bx

# setup image coordinates at each node (assumes parents come before children)
coords = {0: (-1, 0)} # root, -1 before first matching x index
for (n, p, x, y) in E:
    k = np.argmax((X == x).all(axis=1))
    _, c = coords[p]
    coords[n] = (k, c + offsets[n])

# setup the figure
pt.figure(figsize=(8, 6))

# draw the edges
for (n, p, x, y) in E:
    nk, nc = coords[n]
    pk, pc = coords[p]
    if nc == pc:
        pt.plot([pk, nk],[pc, nc],  'k-', zorder=-100)
    else:
        a = mp.Arc((nk, pc), 2*(nk-pk), 2*(nc-pc), angle=0., theta1=90., theta2=180., color='k')
        pt.gca().add_patch(a)

# draw the nodes
for n, (nk, nc) in coords.items():
    pt.gca().add_patch(mp.Circle((nk, nc), rd, ec='k', color='w', fill=True))
    if n < 10:
        pt.text(nk - .33*rd, nc - .5*rd, str(n), fontsize=fs)
    else:
        pt.text(nk - .67*rd, nc - .5*rd, str(n), fontsize=fs)

# draw the vectors
pt.text(-1 - .25*bx, bx*(-N/2 - 3) - .25*bx, '$x_n$', fontsize=fs, color='k')
for k, n in it.product(range(X.shape[0]), range(X.shape[1])):
    if X[k,n] == +1:
        s, ec, fc = '+', 'k', 'w'
    else:
        s, ec, fc = '$-$', 'w', 'k'
    x, y = k, bx*(-n - 3.5)
    pt.gca().add_patch(mp.Rectangle((x-bx/2, y-bx/2), bx, bx, ec='k', fc=fc, fill=True))
    pt.text(x - .25*bx, y - .25*bx, s, fontsize=fs, color=ec)

# draw the labels
pt.text(-1 - .25*bx, -1.5*rd - .5*bx - .25*bx, '$y_n$', fontsize=fs, color='k')
for (n, p, x, y) in E:
    nk, nc = coords[n]
    if y == +1:
        s, ec, fc = '+', 'k', 'w'
    else:
        s, ec, fc = '$-$', 'w', 'k'
    x, y = nk, nc - 1.5*rd - .5*bx
    pt.gca().add_patch(mp.Rectangle((x-bx/2, y-bx/2), bx, bx, ec='k', fc=fc, fill=True))
    pt.text(x - .25*bx, y - .25*bx, s, fontsize=fs, color=ec)

pt.axis('equal')
pt.axis('off')
pt.tight_layout()
pt.savefig("counterexample_tree.pdf")
pt.show()

