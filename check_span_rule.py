def check_span_rule(X, Y, B, W, solver, verbose=False):

    assert len(Y) == len(np.unique(Y, axis=0))
    assert (np.sign(Y) == Y).all()
    assert (np.fabs(Y) == 1).all()

    print("Building tree...")
    node_data = []
    V, E = {(): 0}, []
    for i in range(len(Y)):
        D = ()
        for k in np.flatnonzero(B[i]):
            p = V[D]
            D += ( (k, Y[i,k]), )
            if D not in V:
                V[D] = len(V)
                node_data.append( (p, D, i) )
            n = V[D]
            E.append( (n, p, X[k]) )

    for p, D, i in node_data:
        kn, yn = zip(*D)
        assert (np.sign(W[i] @ X[kn]) == yn).all()
        assert (Xp == Xn[:-1]).all()
        assert (yp == yn[:-1]).all()

    print("Building linear program...")

    ## variables
    u = cp.Variable((len(V), N)) # weight vector per node
    𝛾 = cp.Variable(len(E)) # gamma per spanning tree edge

    ## data constraints
    sample_constraints = [
        u[n] @ (Xn * yn) >= 1
        for n, (p, kk, i, Xk, yk) in enumerate(nodes)
        if p is not None] # no constraints on root

    ## span constraints
    span_constraints = [
        u[n] == u[p] + 𝛾[e] * x
        for e, (n, p, x) in enumerate(E)]

    ## objective to bound problem
    print("Building objective...")
    c = np.stack([
        (Xn * yn).mean(axis=1)
        for (p, kk, i, Xk, yk) in nodes
        if p is not None])
    objective = cp.Minimize(cp.sum(cp.multiply(w[1:], c)))

    constraints = sample_constraints + span_constraints




