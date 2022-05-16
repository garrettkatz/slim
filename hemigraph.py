import os, sys
import itertools as it
from collections import deque
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

N = int(sys.argv[1])
fname = f"hemigraph_{N}.npz"

X = np.array(tuple(it.product((-1, +1), repeat=N))).T
Nh = 2**(N-1) # faster, more numerically stable linprog without antiparallel data
Xh = X[:,:Nh]

if not os.path.exists(fname):

    def representative(w): return tuple(np.sort(np.fabs(w).astype(int)))

    hemis = []
    depths = []
    boundaries = []
    weights = []
    anchors = []
    reprs = set()

    h0 = Xh[-1] # canonical w0 will be last row of identity
    frontier = deque([(h0,0)])
    while len(frontier) > 0:

        h, depth = frontier.popleft()
        if tuple(h) in hemis: continue

        print(f"{len(hemis)}: |frontier| = {len(frontier)}")

        # calculate interior point
        result = linprog(
            c = Xh @ h,
            A_ub = -(Xh * h).T,
            b_ub = -np.ones(Nh),
            bounds = (None, None),
        )
        w = result.x
        assert (np.sign(w @ Xh) == h).all() # should be true when obtained from boundary flip

        # calculate boundary
        boundary = np.zeros(Nh, dtype=bool)
        neighbors = []
        for k in range(Nh):

            # search for feasible boundary point
            others = list(range(k)) + list(range(k+1, Nh))
            result = linprog(
                c = Xh[:,others] @ h[others],
                A_eq = Xh[:,k].reshape(1,-1),
                b_eq = np.zeros(1),
                A_ub = -(Xh[:,others] * h[others]).T,
                b_ub = -np.ones(Nh-1),
                bounds=(None, None))
            wp = result.x

            # check boundary condition
            # boundary[k] = ((wp @ Xh[:,others]) * h[others] >= 1).all() and np.fabs(wp @ Xh[:,k]) * (N-2) <= 1
            boundary[k] = ((wp @ Xh[:,others]) * h[others] >= .99).all() and np.fabs(wp @ Xh[:,k]) <= .01
            if not boundary[k]: continue # not a boundary

            # flip h over boundary
            hf = h.copy()
            hf[k] *= -1
            neighbors.append(hf)

        boundaries.append(boundary)

        # calculate canonical
        w = np.linalg.lstsq(Xh[:,boundary].T, h[boundary], rcond=None)[0]
        w = w.round().astype(int)
        assert (np.sign(w @ Xh) == h).all()

        hemis.append(tuple(h))
        depths.append(depth)
        weights.append(w)

        # check for new representative
        w_rep = representative(w)
        anchors.append(w_rep not in reprs)
        if not anchors[-1]: continue

        reprs.add(w_rep)
        for hf in neighbors: frontier.append((hf, depth+1))

    # # symmetric region generation
    # h0 = Xh[-1] # canonical w0 will be last row of identity
    # frontier = [h0]
    # for depth in it.count():
    #     if len(frontier) == 0: break
    #     print(f"depth {depth}: |hemis| = {len(hemis)}")

    #     new_frontier = []
    #     new_reprs = set(reprs)
    #     for h in frontier:

    #         if tuple(h) in hemis: continue 

    #         # calculate interior point
    #         result = linprog(
    #             c = Xh @ h,
    #             A_ub = -(Xh * h).T,
    #             b_ub = -np.ones(Nh),
    #             bounds = (None, None),
    #         )
    #         w = result.x
    #         assert (np.sign(w @ Xh) == h).all() # should be true when obtained from boundary flip
    
    #         hemis.append(tuple(h))
    #         depths.append(depth)
    
    #         # calculate boundary
    #         boundary = np.zeros(Nh, dtype=bool)
    #         neighbors = []
    #         for k in range(Nh):
    
    #             # search for feasible boundary point
    #             others = list(range(k)) + list(range(k+1, Nh))
    #             result = linprog(
    #                 c = Xh[:,others] @ h[others],
    #                 A_eq = Xh[:,k].reshape(1,-1),
    #                 b_eq = np.zeros(1),
    #                 A_ub = -(Xh[:,others] * h[others]).T,
    #                 b_ub = -np.ones(Nh-1),
    #                 bounds=(None, None))
    #             wp = result.x
    
    #             # check boundary condition
    #             # boundary[k] = ((wp @ Xh[:,others]) * h[others] >= 1).all() and np.fabs(wp @ Xh[:,k]) * (N-2) <= 1
    #             boundary[k] = ((wp @ Xh[:,others]) * h[others] >= .99).all() and np.fabs(wp @ Xh[:,k]) <= .01
    #             if not boundary[k]: continue # not a boundary
    
    #             # flip h over boundary
    #             hf = h.copy()
    #             hf[k] *= -1
    #             neighbors.append(hf)
    
    #         boundaries.append(boundary)
    
    #         # calculate canonical
    #         w = np.linalg.lstsq(Xh[:,boundary].T, h[boundary], rcond=None)[0]
    #         w = w.round().astype(int)
    #         assert (np.sign(w @ Xh) == h).all()
    #         weights.append(w)
    
    #         # save new representatives
    #         w_rep = representative(w)
    #         if w_rep in reprs: continue
    #         new_reprs.add(w_rep)
    
    #         # queue neighbors
    #         new_frontier.extend(neighbors)

    #     # advance frontier
    #     frontier = new_frontier
    #     reprs = new_reprs
    
    hemis = np.stack(hemis)
    depths = np.array(depths)
    boundaries = np.stack(boundaries)
    weights = np.stack(weights)
    anchors = np.array(anchors)
    reprs = np.array(list(reprs))
    
    np.savez(fname, hemis=hemis, depths=depths, boundaries=boundaries, weights=weights, anchors=anchors, reprs=reprs)

else:

    npz = np.load(fname)
    hemis, depths, boundaries, weights, anchors, reprs = npz["hemis"], npz["depths"], npz["boundaries"], npz["weights"], npz["anchors"], npz["reprs"]

print(hemis)
print(weights)
print(reprs)

print(f"{len(reprs)} region classes, {len(hemis)} regions")

import matplotlib.pyplot as pt
import networkx as nx

G = nx.Graph()
for n,(w,h) in enumerate(zip(weights, hemis)):
    G.add_node(n, w=w, h=h)
for n,h in enumerate(hemis):
    for k in np.flatnonzero(boundaries[n]):
        hf = h.copy()
        hf[k] *= -1
        matches = (hemis == hf).all(axis=1)
        m = matches.argmax()
        if matches[m]: G.add_edge(n, m, x=Xh[:,k])

# # biggest wX difference between neighbors?
# for (n,m) in G.edges:
#     print(f"{n,m}: max delta wX = {np.fabs(weights[n] @ Xh - weights[m] @ Xh).max()}")
# input('..')

# # distribution of wX for each anchor
# for i,a in enumerate(np.flatnonzero(anchors)):
#     wX = np.fabs(weights[a] @ Xh)
#     distfreqs = (np.arange(wX.max()+1).reshape(-1,1) == wX).sum(axis=1)
#     print(i, distfreqs)
#     pt.bar(np.arange(wX.max()+1) + i/anchors.sum(), distfreqs, width=1/anchors.sum(), label=str(i), align="edge")
# pt.xlabel("|wx|")
# pt.ylabel("frequency")
# pt.legend()
# pt.show()

# # quick formula for number of boundaries?
# for i,a in enumerate(np.flatnonzero(anchors)):
#     w = weights[a]
#     uni = np.unique(w)
#     counts = (uni.reshape(-1,1) == w).sum(axis=1)
#     boundy = np.array([np.arange(1,c+1).prod() if u != 0 else 2**c for u,c in zip(uni,counts)]).prod()
#     # boundy = (2**counts).prod()
#     print(i, boundaries[a].sum(), w, boundy)

pos = nx.kamada_kawai_layout(G)
# pos = nx.shell_layout(G, nlist=[np.flatnonzero(depths == d) for d in range(depths.max()+1)])
# pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)

for (n,m) in G.edges:
    xn, yn = pos[n]
    xm, ym = pos[m]
    pt.plot([xn, xm], [yn, ym], 'k-')
    pt.text((xn+xm)/2, (yn+ym)/2, str(G.edges[n,m]["x"]), rotation=45)
for n,(w,h,a) in enumerate(zip(weights, hemis, anchors)):
    marker, color = 'or' if a else '.k'
    pt.plot(*pos[n], marker=marker, color=color)
    pt.text(*pos[n], s=str(w))
    # G.add_node(n, w=w, h=h)

# nx.draw_networkx(G, pos=pos)

# shells = [np.flatnonzero(depths == d) for d in range(depths.max()+1)]
# pos = nx.shell_layout(G, nlist=shells)
# nx.draw_networkx(G, pos=pos, with_labels=True)

# nx.draw_circular(G, with_labels=True)

# nx.draw_planar(G, with_labels=True)

pt.show()

