import numpy as np

def graph_fitness(learning_rule, w0, X, Yc, Ac, eps=1, verbose=False):

    region_loss = 0.
    match_loss = 0.
    W = {}
    
    for N in X.keys():

        # organize adjacencies by region
        A = {}
        for (i,j,k) in Ac[N]:
            if i not in A: A[i] = []
            A[i].append((j, k))

        if verbose: print(f"{N} starting, {len(A)} regions")

        # init graph search
        explored = {}
        queue = [(0, w0[N])]

        # run graph search
        while len(queue) > 0:
            i, w = queue.pop()
            if verbose: print(f" {N}, {i} region popped")

            # don't repeat work; previously explored should match
            if i in explored:
                diff = np.sum((w - explored[i])**2)
                if verbose: print(f" {N}, {i} explored, diff = {diff}")
                match_loss += diff
                continue

            explored[i] = w

            # check region constraint
            dots = (w * (X[N] * Yc[N][i]).T).sum(axis=1) # should be > eps
            violation = np.fabs(np.minimum(dots - eps, 0)).max()
            region_loss += violation
            if verbose: print(f" {N}, {i} new, violation = {violation}")

            # queue up children
            for (j,k) in A[i]:
                w_new = learning_rule(w, X[N][:,k], Yc[N][j,k])
                queue.append((j, w_new))

        # save W results
        W[N] = np.array([explored[i] for i in range(len(explored))])

    return region_loss, match_loss, W

if __name__ == "__main__":

    import pickle as pk

    Ns = [3, 4, 5]
    Yc, W, X, Ac = {}, {}, {}, {}

    for N in Ns:

        # load canonical hemis
        ltms = np.load(f"ltms_{N}_c.npz")
        Yc[N], W[N], X[N] = ltms["Y"], ltms["W"], ltms["X"]

        with open(f"adjs_{N}_jc.npz", "rb") as f: Ac[N] = pk.load(f)

    def perceptron_rule(w, x, y):
        return w + (y - np.sign((w * x).sum())) * x

    def constant_rule(w, x, y):
        return w

    def hebbian_rule(w, x, y):
        return w + y * x # / len(x)
        # return (w + y * x) / (len(x) - 1)

    w0 = {N: W[N][0] for N in Ns}

    region_loss, match_loss, W = graph_fitness(perceptron_rule, w0, X, Yc, Ac, eps=0.5, verbose=True)
    print('perceptron', region_loss, match_loss, '\n')
    print(W[3])
    print()

    region_loss, match_loss, W = graph_fitness(hebbian_rule, w0, X, Yc, Ac, eps=1.0, verbose=True)
    print('hebbian', region_loss, match_loss, '\n')
    print(W[3])
    print()

    region_loss, match_loss, W = graph_fitness(constant_rule, w0, X, Yc, Ac, eps=0.5, verbose=True)
    print('constant', region_loss, match_loss, '\n')
    print(W[3])
    print()

