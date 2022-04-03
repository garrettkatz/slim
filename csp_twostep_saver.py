"""
Use to find maximal M for given N
"""
from scipy.special import comb
import pickle as pk
import numpy as np
import itertools as it

N = 4 # number of neurons
M = 4 # number of key-value pairs
ortho = True
justone_soln = False
do_check = True
check_first = False
check_all = False
check_random = True
do_csp = True
verbose = True
revert_onestep = False
save_depth = 1

save_mod = M**(M - save_depth)

orth = "orth" if ortho else ""

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
            solns[vidx].append((hidx, m1, m2))
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

shattered = True
kidx = tuple(range(M))
if N == 4 and M == 4 and ortho: kidx = (0, 3, 5, 6)

if do_csp:
    
    hidx = [np.arange(2**N) for _ in range(M)]
    m1 = [np.arange(H.shape[0]) for _ in range(N)]
    m2 = [np.arange(H.shape[0]) for _ in range(N)]

    # constrain one layer to identity to encapsulate onestep
    if revert_onestep:
        ww = np.array(weights).reshape(-1,N)
        m1 = [np.array(np.isclose(np.eye(N)[i], ww).all(axis=1).argmax()) for i in range(N)]
        
    for v,vidx in enumerate(it.product(kidx, repeat=M)):
    
        if v % save_mod == 0: solns = {}
        solns[vidx] = []
    
        j = -1
        k1 = -1
        solved = constrain(kidx, vidx, j, k1, hidx, m1, m2, solns, justone_soln)
        shattered = shattered and solved
        if not solved: break
    
        if verbose: print(f"{v} of {M**M}", vidx, f"{len(solns[vidx])} solns stored")
    
        if (v+1) % save_mod == 0:
            vlead = "_".join(map(str, vidx[:save_depth]))
            if revert_onestep:
                fname = f"solns/N{N}M{M}S1{orth}_{vlead}"
            else:
                fname = f"solns/N{N}M{M}{orth}_{vlead}"
            with open(fname, "wb") as f: pk.dump(solns, f)
    
    print("kidx shatters:", shattered)

if do_check and shattered:
    print("checking solns")
    least_solns, largest_solns = np.inf, 0
    all_solved = True
    for v,vidx in enumerate(it.product(kidx, repeat=M)):

        if v % save_mod == 0:
            vlead = "_".join(map(str, vidx[:save_depth]))
            if revert_onestep:
                fname = f"solns/N{N}M{M}S1{orth}_{vlead}"
            else:
                fname = f"solns/N{N}M{M}{orth}_{vlead}"
            with open(fname, "rb") as f: solns = pk.load(f)

        if len(solns[vidx]) == 0:
            all_solved = False
            break

        print(f"{v} of {M**M}", vidx, f"{len(solns[vidx])} solns")
        largest_solns = max(largest_solns, len(solns[vidx]))
        least_solns = min(least_solns, len(solns[vidx]))

        for hidx, m1, m2 in solns[vidx]:
            # check first m1/m2 soln
            if check_first:
                m1 = [m1[i][0] for i in range(N)]
                m2 = [m2[i][0] for i in range(N)]
                W1 = np.concatenate([weights[i] for i in m1], axis=0)
                W2 = np.concatenate([weights[i] for i in m2], axis=0)
                Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
                assert np.allclose(Y, V[:,vidx])

            # check random m1/m2 soln
            elif check_random:
                m1 = [np.random.choice(m1[i]) for i in range(N)]
                m2 = [np.random.choice(m2[i]) for i in range(N)]
                W1 = np.concatenate([weights[i] for i in m1], axis=0)
                W2 = np.concatenate([weights[i] for i in m2], axis=0)
                Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
                assert np.allclose(Y, V[:,vidx])

            # check all m1/m2 solns
            elif check_all:
                for i1 in it.product(*m1):
                    W1 = np.concatenate([weights[i] for i in i1], axis=0)
                    assert np.allclose(V[:,hidx], np.sign(W1 @ V[:,kidx]))
                for i2 in it.product(*m2):
                    W2 = np.concatenate([weights[i] for i in i2], axis=0)
                    assert np.allclose(V[:,vidx], np.sign(W2 @ V[:,hidx]))

    if all_solved:
        print("confirmed kidx shatters")

    print(f"{least_solns}-{largest_solns} solns")

