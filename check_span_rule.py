import numpy as np
import cvxpy as cp

def check_span_rule(X, Y, B, W, solver, verbose=False):

    N = X.shape[1]
    assert (np.fabs(X) == 1).all()
    assert (np.fabs(Y) == 1).all()
    assert len(X) == len(np.unique(X, axis=0))
    assert len(Y) == len(np.unique(Y, axis=0))

    print("Building the tree...")

    V, E = {(): (0, 0)}, []
    for i in range(len(Y)):
        D = ()
        for k in np.flatnonzero(B[i]):
            p, _ = V[D]
            D += ((k, Y[i,k]),)
            if D not in V:
                n = len(V)
                V[D] = (n, i)
                E.append( (n, p, X[k], Y[i,k]) )

    D = [(X[[]], np.empty(0), 0)]
    for (Dn, (n, i)) in V.items():
        if n == 0: continue
        ks, y = map(np.int64, zip(*Dn))
        D.append((X[ks], y, i))

    print("Checking the tree...")

    for (Xn, yn, i) in D:
        assert (np.sign(W[i] @ Xn.T) == yn).all()
    for (n, p, x, y) in E:
        Xn, yn, _ = D[n]
        Xp, yp, _ = D[p]
        assert (Xp == Xn[:-1]).all()
        assert (yp == yn[:-1]).all()

    print("Running the linear program...")

    u = cp.Variable((len(D), N))
    g = cp.Variable(len(E))

    span_constraints = [
        u[n] == (u[p] + g[e] * x)
        for e, (n, p, x, _) in enumerate(E)]

    data_constraints = [
        u[n] @ (Xn.T * yn) >= 1
        for n, (Xn, yn, _) in enumerate(D) if n > 0]

    c = np.stack([
        (Xn.T * yn).sum(axis=1)
        for n, (Xn, yn, _) in enumerate(D) if n > 0])

    constraints = span_constraints + data_constraints
    objective = cp.Minimize(cp.sum(cp.multiply(u[1:],c)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    return (
        problem.status,
        u.value,
        g.value,
        D, E
    )


