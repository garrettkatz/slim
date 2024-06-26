"""
Use to find maximal M for given N
"""
from scipy.special import comb
import pickle as pk
import numpy as np
import itertools as it

N = 3 # number of neurons
M = 4 # number of key-value pairs
justone_kidx=True
justone_soln=True
store_solns=False
do_check=True
verbose=True

V = np.array(tuple(it.product((-1, 1), repeat=N))).T
print("V.shape:", V.shape) # (num neurons N, num verticies 2**N)

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, H = pk.load(f)
print("H.shape:", H.shape) # (num_dichotomies, num vertices = 2**N)

def propagate_contraints(m1, m2, k, h, v):
    for i in range(N):
        m1[i] = m1[i][H[m1[i], k] == V[i, h]]
        if len(m1[i]) == 0: return False
        m2[i] = m2[i][H[m2[i], h] == V[i, v]]
        if len(m2[i]) == 0: return False
    return True

def constrain(kidx, vidx, j, k, hidx, m1, m2, solns, justone):
    if j >= 0:
        # copy for recursion
        hidx, m1, m2 = list(hidx), list(m1), list(m2)
        hidx[j] = hidx[j][k]
        # propagate m constraints
        for i in range(N):
            m1[i] = m1[i][H[m1[i], kidx[j]] == V[i, hidx[j]]]
            if len(m1[i]) == 0: return False # unsatisfiable
            m2[i] = m2[i][H[m2[i], hidx[j]] == V[i, vidx[j]]]
            if len(m2[i]) == 0: return False # unsatisfiable
        # feas = propagate_contraints(m1, m2, kidx[j], hidx[j], vidx[j])
        # if not feas: return False
        # base case; only reached if satisfiable
        if j == M-1:
            if store_solns:
                solns[kidx][vidx].append((hidx, m1, m2))
            return True
        # else: # seems slower
        #     # propagate j+ constraints
        #     for i, jp in it.product(range(N), range(j+1,M)):
        #         match = (H[m2[i][:,np.newaxis], hidx[jp]] == V[i, vidx[jp]])
        #         hidx[jp] = hidx[jp][match.any(axis=0)]
        #         m2[i] = m2[i][match.any(axis=1)]
        #         if 0 in (len(hidx[jp]), len(m2[i])):
        #             return False # unsatisfiable
    # # also slower
    # for i in range(N):
    #     hidx[j+1] = hidx[j+1][(H[m2[i][:,np.newaxis], hidx[j+1]] == V[i, vidx[j+1]]).any(axis=0)]
    # recurse
    any_solved = False
    for k1 in range(len(hidx[j+1])):
        solved = constrain(kidx, vidx, j+1, k1, hidx, m1, m2, solns, justone)
        any_solved = any_solved or solved
        if solved and justone: break
    return any_solved

kidxs = []
solns = {}
any_shattered = False
for k,kidx in enumerate(it.combinations(range(2**N), M)):
    # kidx is unshatterable if it contains both +/- v
    if any((2**N - j - 1) in kidx for j in kidx): continue
    solns[kidx] = {}
    shattered = True

    hidx = [np.arange(2**N) for _ in range(M)]
    m1 = [np.arange(H.shape[0]) for _ in range(N)]
    m2 = [np.arange(H.shape[0]) for _ in range(N)]
        
    for v,vidx in enumerate(it.product(kidx, repeat=M)):

        if store_solns:
            solns[kidx][vidx] = []

        j = -1
        k1 = -1
        solved = constrain(kidx, vidx, j, k1, hidx, m1, m2, solns, justone_soln)
        shattered = shattered and solved
        if not solved: break

        if verbose: print(kidx, f"{v} of {M**M}", vidx,
            f"{len(solns[kidx][vidx])} solns stored" if store_solns else "")

    print(f"{k} of {comb(V.shape[1], M, exact=True)} kidxs: {kidx} shattered={shattered}...")

    any_shattered = any_shattered or shattered
    
    if shattered and justone_kidx: break # only find one shattered kidx

    # actually, if col permutation permutes with forward pass, either all or none kidxs shatter
    break

print("Any kidxs shatter:", any_shattered)

if do_check and store_solns and any_shattered:
    print("shattered kidxs:")
    num_shattered = 0
    largest_solns = 0
    for kidx in sorted(solns.keys()):
        all_solved = True
        for v,vidx in enumerate(it.product(kidx, repeat=M)):
            if len(solns[kidx][vidx]) == 0:
                all_solved = False
                break
            print(kidx, f"{v} of {M**M}", vidx, f"{len(solns[kidx][vidx])} solns")
            largest_solns = max(largest_solns, len(solns[kidx][vidx]))
            for hidx, m1, m2 in solns[kidx][vidx]:
                # # check first m1/m2 soln
                # m1 = [m1[i][0] for i in range(N)]
                # m2 = [m2[i][0] for i in range(N)]
                # W1 = np.concatenate([weights[i] for i in m1], axis=0)
                # W2 = np.concatenate([weights[i] for i in m2], axis=0)
                # Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
                # assert np.allclose(Y, V[:,vidx])

                # check random m1/m2 soln
                m1 = [np.random.choice(m1[i]) for i in range(N)]
                m2 = [np.random.choice(m2[i]) for i in range(N)]
                W1 = np.concatenate([weights[i] for i in m1], axis=0)
                W2 = np.concatenate([weights[i] for i in m2], axis=0)
                Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
                assert np.allclose(Y, V[:,vidx])

                # # check all m1/m2 solns
                # for i1 in it.product(*m1):
                #     W1 = np.concatenate([weights[i] for i in i1], axis=0)
                #     assert np.allclose(V[:,hidx], np.sign(W1 @ V[:,kidx]))
                # for i2 in it.product(*m2):
                #     W2 = np.concatenate([weights[i] for i in i2], axis=0)
                #     assert np.allclose(V[:,vidx], np.sign(W2 @ V[:,hidx]))

        if all_solved:
            print(kidx, " shatterable")
            num_shattered += 1
    # print(f"{num_shattered} of {comb(V.shape[1], M, exact=True)} shattered kidxs")
    print(f"{num_shattered} of {len(solns)} shattered kidxs")
    print(f"{largest_solns} solns at most")

