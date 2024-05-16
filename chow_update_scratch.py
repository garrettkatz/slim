import sys
import pickle as pk
import itertools as it
import numpy as np
import cvxpy as cp

np.set_printoptions(linewidth=1000)

N = int(sys.argv[1])
solver = 'GUROBI'

with np.load(f"chow_regions_{N}_{solver}.npz") as regions:
    X, Y, B, W, C = (regions[key] for key in ("XYBWC"))

with open(f"canon_adjacencies_{N}.pkl","rb") as f: A = pk.load(f)

print("i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], W[j] - W[i]")
# for (i,j,k) in sorted(A):
for (i,j,k) in [(921,926,60)]:
    print(i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], (W[j] - W[i]).round(1))
    print('  C[i]', C[i])
    print('  C[j]', C[j])

    ab = np.linalg.lstsq(np.stack((W[i], X[k])).T, W[j], rcond=None)[0]
    print('  span', ab, np.stack((W[i], X[k])).T @ ab)

    ui = cp.Variable(N)
    uj = cp.Variable(N)
    gam = cp.Variable(1)
    prob = cp.Problem(cp.Minimize(cp.sum(ui) + cp.sum(uj)), constraints=[
        ui @ (X.T * Y[i]) >= 1,
        uj @ (X.T * Y[j]) >= 1,
        ui == uj + gam * X[k],
    ])
    prob.solve(solver=solver, verbose=False)
    print('span cp', prob.status)
    print('ui', ui.value)
    print('uj', uj.value)

    ### update attempts

    # ## projection into linear space of chow constraints and required X[k] output
    # ## fails the assertion in N=4

    # # chow equality constraints
    # A = np.eye(N) - np.eye(N,k=1)
    # idx = np.flatnonzero(C[j,:-1] == C[j,1:])
    # if C[j,-1] == 0: idx = np.append(idx, N-1)

    # # required X[k] output
    # A = np.concatenate((A[idx], X[[k]]), axis=0)

    # # 0 on chow, +/- 1 on output
    # b = np.zeros(len(A))
    # b[-1] = Y[j,k]

    # # solve linear equalities for smallest  delta d, where
    # # A(wi + d) = b -> Ad = b - Awi
    # d = np.linalg.lstsq(A, b - A @ W[i], rcond=None)[0]
    # wj = (W[i] + d).round(1)

    # ## satisfy X[k] and maintain x' with hamming distance 1 from X[k]
    # ## satisfies assertion on N=4 but not chow symmetry
    # ## fails the assertion on N=5

    # # ham dist 1 vectors
    # A = X[k] * (1 - 2*np.eye(N))
    # A[0] = X[k] # except don't flip first bit in half-cube

    # # maintain same dots as W[i]
    # b = A @ W[i]
    # b[0] *= -1 # except X[k] must flip

    # # solve it: A wj = b
    # wj = np.linalg.lstsq(A, b, rcond=None)[0].round()

    # ## satisfy chow, and for remaining DoFs, flip bits for lowest w's
    # ## fails assert at N=4 and has issues, like invariance when low W[i] are 0
    # ## should refine this to account for chow symmetries, and try to iterate bit flips in order of changing effect on dots with w

    # # chow constraints
    # A_chow = np.eye(N) - np.eye(N,k=1)
    # idx = np.flatnonzero(C[j,:-1] == C[j,1:])
    # if C[j,-1] == 0: idx = np.append(idx, N-1)
    # A_chow = A_chow[idx]
    # b_chow = np.zeros(len(A_chow))

    # # low bit constraints
    # A_low = np.empty((N-len(A_chow), N))
    # b_low = np.empty(N - len(A_chow))
    # for n in range(len(A_low)):
    #     A_low[n] = X[k]
    #     A_low[n,-n-1:] *= -1
    #     b_low[n] = A_low[n] @ W[i]

    # A = np.concatenate((A_chow, A_low), axis=0)
    # b = np.concatenate((b_chow, b_low))

    # # solve it: A wj = b
    # wj = np.linalg.lstsq(A, b, rcond=None)[0].round()

    # ## satisfy chow, canonical, projected dot, new dot, and minimize weight
    # ## satisfies assert at N=4 but fails N=5, refine projected dot constraint?
    
    # u = cp.Variable(N)

    # # enforce chow symmetries
    # n = np.flatnonzero(C[j,:-1] == C[j,1:])
    # chow_constraints = [u[n] == u[n+1]]
    # if C[j,-1] == 0: chow_constraints.append(u[-1] == 0)

    # # stay canonical
    # canonical_constraints = [u[-1] >= 0, u[:-1] >= u[1:]]

    # # projected dot
    # wiP = W[i] - (W[i] @ X[k]) * X[k] / N
    # # dot_constraints = [wiP @ u >= 3.75, X[k] @ u >= 1]
    # # dot_constraints = [wiP @ u >= ((N+1)/N - 2**(2-N)), X[k] @ u >= 1]
    # tight = (2**(N-1) * (W[i]**2).sum() - (W[i] @ X[k])**2) ** .5 / (2**(N-1)) + (1/N - 1/(2**(N-1))) * np.fabs(W[i] @ X[k])
    # tight = 1
    # print('  tight  ',tight)
    # dot_constraints = [wiP @ u >= tight, X[k] @ u >= 1]

    # constraints = chow_constraints + canonical_constraints + dot_constraints
    # objective = cp.Minimize(cp.sum(u))
    # problem = cp.Problem(objective, constraints)
    # problem.solve(solver=solver, verbose=False)

    # wj = u.value

    # print('  wiP    ',wiP)

    ## satisfy chow, canonical, non-neg new dot, at most unit norm, and maximize projected dot
    ## satisfies assert for N<8 but fails N=8, where span broke down
    ## the first problem case is a pair where both have no symmetries (no chow params equal to eachother or zero)
    
    u = cp.Variable(N)

    # enforce chow symmetries
    chow_constraints = []
    n = np.flatnonzero(C[j,:-1] == C[j,1:])
    if len(n) > 0: chow_constraints.append(u[n] == u[n+1])
    if C[j,-1] == 0: chow_constraints.append(u[-1] == 0)

    # stay canonical
    canonical_constraints = [u[-1] >= 0, u[:-1] >= u[1:]]

    # non-neg new dot
    dot_constraints = [X[k] @ u >= 0]

    # at most unit norm
    norm_constraints = [cp.norm(u) <= 1]
    # norm_constraints = [cp.norm(u,1) <= 1]

    # projected dot
    wiP = W[i] - (W[i] @ X[k]) * X[k] / N
    objective = cp.Maximize(wiP @ u)

    constraints = chow_constraints + canonical_constraints + dot_constraints + norm_constraints
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    wj = u.value

    print('  wiP    ',wiP)

    # ## try active set on nearby x', does not look promising
    
    # eps = 1
    # wi, xk = W[i], X[k]

    # # # this produces s(v) = -xk
    # # b = -1 / (N - (xk @ wi)**2 / (wi @ wi + eps**2) )
    # # v = np.sign( b * (xk - wi * (xk @ wi) / (wi @ wi + eps**2) ) )

    # # this produces correct nearby neighbors in N=4
    # v, t = cp.Variable(N), cp.Variable(1)
    # constraints = [
    #     v @ xk == -1,
    #     t*eps - wi @ v <= 0,
    #     -t <= v, v <= t,
    #     -(N-2) <= xk @ v, xk @ v <= N-2,
    # ]
    # objective = cp.Maximize(t*eps - wi @ v)
    # problem = cp.Problem(objective, constraints)
    # problem.solve(solver=solver, verbose=False)

    # x1 = np.sign(v.value)
    # l1 = (eps - wi @ x1) / (xk @ x1)
    # w1 = wi + l1 * xk

    # d1 = xk - (xk @ x1) * x1 / N

    # print('  v1     ', v.value)
    # print('  x1     ', x1)
    # print('  l1     ', l1)
    # print('  w1     ', w1)
    # print('  wi x1  ', wi @ x1)
    # print('  w1 x1  ', w1 @ x1)
    # print('  w1 xk  ', w1 @ xk)
    # print('  d1     ', d1)
    # print('  ***')

    # # repeat on d1 instead of xk
    # v, t = cp.Variable(N), cp.Variable(1)
    # constraints = [
    #     v @ d1 == -1,
    #     t*eps - w1 @ v <= 0,
    #     -t <= v, v <= t,
    #     # -(N-2) <= xk @ v, xk @ v <= N-2,
    #     -(N-2) <= x1 @ v, x1 @ v <= N-2,
    # ]
    # objective = cp.Maximize(t*eps - w1 @ v)
    # problem = cp.Problem(objective, constraints)
    # problem.solve(solver=solver, verbose=False)

    # x2 = np.sign(v.value)
    # l2 = (eps - w1 @ x2) / (d1 @ x2)
    # w2 = w1 + l2 * d1

    # # d2 normal to both x1 and x2, in direction of xk
    # g2 = x2 - (x2 @ x1) * x1 / N
    # d2 = xk - (xk @ x1) * x1 / N - (xk @ g2) * g2 / (g2 @ g2)

    # print('  v2     ', v.value)
    # print('  x2     ', x2)
    # print('  l2     ', l2)
    # print('  w2     ', w2)
    # print('  wi x2  ', wi @ x2)
    # print('  w1 x2  ', w1 @ x2)
    # print('  w2 x2  ', w2 @ x2)
    # print('  w2 x1  ', w2 @ x1)
    # print('  w2 xk  ', w2 @ xk)
    # print('  d2     ', d2)
    # print('  ***')

    # # repeat on d2 instead of xk
    # v, t = cp.Variable(N), cp.Variable(1)
    # constraints = [
    #     v @ d2 == -1,
    #     t*eps - w2 @ v <= 0,
    #     -t <= v, v <= t,
    #     # -(N-2) <= xk @ v, xk @ v <= N-2,
    #     -(N-2) <= x2 @ v, x2 @ v <= N-2,
    # ]
    # objective = cp.Maximize(t*eps - w2 @ v)
    # problem = cp.Problem(objective, constraints)
    # problem.solve(solver=solver, verbose=False)

    # x3 = np.sign(v.value)
    # l3 = (eps - w2 @ x3) / (d2 @ x3)
    # w3 = w2 + l3 * d2

    # # d3 normal to x1,x2,x3, in direction of xk
    # g3 = x3 - (x3 @ x1) * x1 / N - (x3 @ g2) * g2 / (g2 @ g2)
    # d3 = xk - (xk @ x1) * x1 / N - (xk @ g2) * g2 / (g2 @ g2) - (xk @ g3) * g3 / (g3 @ g3)

    # print('  v3     ', v.value)
    # print('  x3     ', x3)
    # print('  l3     ', l3)
    # print('  w3     ', w3)
    # print('  wi x3  ', wi @ x3)
    # print('  w1 x3  ', w1 @ x3)
    # print('  w3 x3  ', w3 @ x3)
    # print('  w3 x1  ', w3 @ x1)
    # print('  w3 xk  ', w3 @ xk)
    # print('  d3     ', d3)
    # print('  ***')

    # wj = w3

    ### compare result with desired
    print('  wj     ',wj)
    print('  x W[i]   ', (X @ W[i]))
    print('  x W[j]   ', (X @ W[j]))
    print('  x wj   ', (X @ wj))
    print('  s(x wj)', np.sign(X @ wj).astype(int))
    print('  yj     ', Y[j])
    

    if not (np.sign(X @ wj) == Y[j]).all():
        kk = np.flatnonzero(np.sign(X @ wj) != Y[j])
        print('  bad k  ', kk)
        print('  Xb\n', X[kk])
        print(W[i])
        print(W[j])
        print(wj)
        print('  xb @ wj', X[kk] @ wj)
        print('  xb @ W[j]', X[kk] @ W[j])

    assert (np.sign(X @ wj) == Y[j]).all()


