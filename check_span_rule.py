def check_span_rule(X, Y, B, W, solver, verbose=False):

    N = X.shape[1]
    assert (np.fabs(X) == 1).all()
    assert (np.fabs(Y) == 1).all()
    assert len(X) == len(np.unique(X, axis=1))
    assert len(Y) == len(np.unique(Y, axis=0))

    print("Building the tree...")

    V, E = {((), ()): (0, 0)}, []
    for i in range(len(Y)):
        Dk, Dy = (), ()
        for k in np.flatnonzero(B[i]):
            p = V[Dk, Dy]
            Dk += (k,)
            Dy += (Y[i,k],)
            if Dk, Dy not in V:
                n = len(V)
                V[Dk, Dy] = (n, i)
                E.append( (n, p, X[k], Y[i,k]) )

    V = [(X[Dk], np.array(Dy), i)
        for (Dk, Dy), (_, i) in V.items()]

    print("Checking the tree...")

    for (Xn, yn, i) in V:
        assert (np.sign(W[i] @ Xn.T) == yn).all()
    for (n, p, x, y) in E:
        Xn, yn, _ = V[n]
        Xp, yp, _ = V[p]
        assert (Xn == np.append(Xp, x.reshape(1,N)).all()
        assert (yn == np.append(yp, y)).all()

    print("Running the linear program...")

    u = cp.Variable((len(V), N))
    ɣ = cp.Variable(len(E))

    span_constraints = [
        u[n] == u[p] + ɣ[e] * x
        for e, (n, p, x, y) in enumerate(E)]

    data_constraints = [
        u[n] @ (Xn.T * yn) >= 1
        for n, (Xn, yn, _) in enumerate(nodes) if n > 0]

    c = np.stack([
        (Xn.T * yn).mean(axis=1)
        for n, (Xn, yn, _) in enumerate(nodes) if n > 0])

    objective = cp.Minimize(cp.sum(cp.multiply(u[1:], c)))
    constraints = sample_constraints + span_constraints

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    feasible = (problem.status == 'optimal')




