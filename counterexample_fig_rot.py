import itertools as it
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as pt
import matplotlib.patches as mp 
from matplotlib import rcParams
from check_span_rule import *

rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True

solver = "GUROBI"
N = 8

# load the regions and boundaries
with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# the adversarial pair have the same symmetries; are there other pairs like that?
all_chow = Y @ X
all_syms = []
for i, chow in enumerate(all_chow):
    syms = tuple(np.flatnonzero(chow[:-1] == chow[1:]))
    if chow[-1] == 0: syms = syms + (N,)
    all_syms.append(syms)

print(f"{len(set(all_syms))} of {len(all_syms)} unique chow symmetries")    


# sort ascending by number of boundaries
sorter = np.argsort(B.sum(axis=1))

# extract the counterexample
sample = sorter[[215, 503]] # indices after sorting
Y = Y[sample]
B = B[sample]
W = W[sample]

# check chow parameters
chow0 = Y[0] @ X
chow1 = Y[1] @ X
assert not (Y[0] == Y[1]).all()
print('chow0', chow0)
print('chow1', chow1)

# input('..')

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
# pt.show()

pt.close()

# now draw the certificate
Z = [X.T*y for (X,y,_) in D]
A = [[None for _ in range(6)] for _ in range(6)]
for n in range(6,12):
    A[n-6][0] = Z[n].T
    m = n
    while m > 6:
        _, p, x, y = E[m-1]
        print(Z[n].shape, x.shape)
        A[n-6][m-6] = Z[n].T @ x.reshape(-1,1)
        m = p

A = sp.bmat(A).toarray()

# another little LP to get the certificate
s = cp.Variable(len(A))
constraints = [
    (s >= 0),
    (cp.sum(s) >= 1),
    (s @ A == 0)
]
objective = cp.Minimize(cp.sum(s))
problem = cp.Problem(objective, constraints)
problem.solve(solver=solver, verbose=True)
s = s.value
# whole numbers for easier checking
s = (s/np.min(s[s > 0])).round().astype(int)

A = A.astype(int)
sA = s @ A
print("s @ A:", s @ A)
print(f"s @ 1 = {s.sum()}")

fs = 20

pt.figure(figsize=(8,20))
pt.imshow(A, alpha=.5)
pt.imshow(s.reshape(-1,1), extent = (-2.5, -1.5, len(A)-0.5, -0.5), vmin=A.min(), vmax=A.max(), alpha=.5)
pt.imshow(np.ones((len(s),1)), extent = (A.shape[1]+3.5, A.shape[1]+4.5, len(A)-0.5, -0.5), vmin=A.min(), vmax=A.max(), alpha=.5)
pt.imshow(sA.reshape(1,-1), extent = (-.5, A.shape[1]-.5, len(A)+1.5, len(A)+.5), vmin=A.min(), vmax=A.max(), alpha=.5)
# pt.imshow(np.array([[s.sum()]]), extent = (A.shape[1]+3.5, A.shape[1]+4.5, len(A)+1.5, len(A)+.5), vmin=A.min(), vmax=A.max(), alpha=.5)

for (i,j) in it.product(*map(range, A.shape)):
    if A[i,j] == 0: continue
    num = str(A[i,j])
    if A[i,j] > 0: num = " "+num
    if A[i,j] == +1: num = "+"
    if A[i,j] == -1: num = "-"
    pt.text(j-.25,i+.25, num, fontsize=fs)

# pt.text(3.5-.25, -.75, "$u_6$", fontsize=fs)
# for j in range(8, A.shape[1]):
#     pt.text(j-.25, -.75, "$\gamma_{%d}$" % (j-8+7), fontsize=fs)

pt.gca().add_patch(mp.Rectangle((A.shape[1]+.5, -.5), 1.5, A.shape[1], ec='k', fill=False))
pt.text(A.shape[1]+1-.25, 4-.75, "$u_6$", fontsize=fs)
for j in range(8, A.shape[1]):
    pt.text(A.shape[1]+1-.25, j+.25, "$\gamma_{%d}$" % (j-8+7), fontsize=fs)

for i in range(len(s)):
    if s[i] == 0: continue
    pt.text(-2.25,i+.25,str(s[i]), fontsize=fs)

for i in range(len(s)):
    pt.text(A.shape[1]+4-.25, i+.25,"+", fontsize=fs)

pt.gca().add_patch(mp.Rectangle((A.shape[1]+3.5, len(A)+.5), 1, 1, ec='k', fill=False))


pt.text(A.shape[1]/2 - .5, -1.75, "$A$", fontsize=fs)
pt.text(A.shape[1]+1-.25, -1.75, "$v$", fontsize=fs)
pt.text(A.shape[1]+2-.25, -1.75, "$\geq$", fontsize=fs)
pt.text(A.shape[1]+4-.25, -1.75, "$1$", fontsize=fs)
pt.text(-2.25, -1.75, "$s$", fontsize=fs)
pt.text(A.shape[1]/2 - 1, len(A) + 1.3, "$s^T A$", fontsize=fs)
pt.text(A.shape[1]+4-.5, len(A) + 1.3, str(s.sum()), fontsize=fs)

# pt.colorbar()
pt.xlim([-3.5, A.shape[1]+5.5])
pt.ylim([len(A)+2.5, -2.5])
pt.axis("off")
pt.tight_layout()
pt.savefig("certificate.pdf")
pt.show()

