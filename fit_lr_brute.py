"""
Searching for digraph with following properties:
- Every node is a net (m1, m2, plus resulting hidx/vidx)
- Every node has M**2 out-edges, one for each possible vidx[j] -> new_vidx[j] transition
- Every edge must be fit by the LR
- Every possible vidx (M**M) must be covered by at least one node
    This should be satisfied as long as you chase every possible out-edge
"""
import pickle as pk
import itertools as it

N = 2 # number of neurons
M = 2 # number of key-value pairs
save_depth = 0

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

print(len(solns))
print(M**M)

def leading_construct(digraph, node, edge_data, j, k, fit_lr):

def construct(digraph, node, edge_data, fit_lr):
    # all edges must fit
    feas = fit_lr(edge_data)
    if not feas: return False, digraph
    # every node is a net with associated hidx/vidx
    m1, m2, hidx, vidx = node
    if (m1, m2) in digraph: return True, digraph
    # every node has all possible M**2 out-edges
    for j,k in it.product(range(M), repeat=2):
        new_vidx = vidx[:j] + (k,) + vidx[j+1:]
        feas = False
        for new_hidx, mm1, mm2 in solns[new_vidx]:
            for m1jk, m2jk in it.product(it.product(*mm1), it.product(*mm2)):
                new_node = (m1jk, m2jk, new_hidx, new_vidx)
                new_edge_data = edge_data + [(node, new_node)]
                sub_feas, sub_digraph = construct(dict(digraph), new_node, new_edge_data, fit_lr)
                feas = feas or sub_feas
                if feas: break
            if feas: break
        if not feas: return False
        digraph = sub_digraph
    digraph[(m1, m2)] = (hidx, vidx)
    return True, digraph

def fit_lr_mock(edge_data):
    return True

vidx = (0,)*M
feas = False
for s, (hidx, mm1, mm2) in enumerate(solns[vidx]):
    for n, (m1, m2) in enumerate(it.product(it.product(*mm1), it.product(*mm2))):
        print(f"{s},{n} of {len(solns[vidx])}")
        feas = feas or construct({}, (m1, m2, hidx, vidx), [], fit_lr_mock)
        if feas: break
    if feas: break
print("Feas:", feas)

