import numpy as np

"""
Runs a graph search over the canonical region adjacency graphs
Favors rules that respect all region constraints and produce one unique weight vector per region
Can run searches for multiple dimensions N 
Applies learning_rule at each transition, should be a function handle
    w_new = learning_rule(w, x, y)
Starts in region whose dichotomy assigns +1 to all vertices of the half-cube
w0 are initial weights for this region
X[N][:,k]: kth vertex of half-cube in dimension N
Yc[N][i,k]: label for kth vertex in ith canonical dichotomy in dimension N
Ac[N]: list of (i,j,k) adjacencies in dimension N
eps: minimum slack enforced for region constraints
verbose: if True, print graph search progress messages
"""
def graph_fitness(learning_rule, w0, X, Yc, Ac, eps=1, verbose=False):

    # loss when the rule violates a region constraint
    region_loss = 0.

    # loss when the rule produces a different weight vector when transitioning to a previously explored region
    match_loss = 0.

    # Saves resulting weights produced by learning rule
    W = {}

    # Process one N at a time    
    for N in X.keys():

        # organize adjacencies by source region
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

            # pop next region to process
            i, w = queue.pop()
            if verbose: print(f" {N}, {i} region popped")

            # check if this region was already explored by the graph search
            if i in explored:

                # if so, the popped weights should match what they were when it was first explored
                # so accumulate any difference in the match loss
                diff = np.sum((w - explored[i])**2)
                match_loss += diff
                if verbose: print(f" {N}, {i} explored, diff = {diff}")

                # and don't repeat any more work since this region was already explored
                continue

            # mark this region as explored the first time it is popped
            explored[i] = w

            # check region constraints
            dots = (w * (X[N] * Yc[N][i]).T).sum(axis=1)
            # all should have constraint slack >= eps
            violation = np.fabs(np.minimum(dots - eps, 0)).max()
            # accumulate any violation in the region loss
            region_loss += violation
            if verbose: print(f" {N}, {i} new, violation = {violation}")

            # use learning rule to push new weights for adjacent regions into the queue
            for (j,k) in A[i]:
                # transitioning to new region j, w_new should fit dichotomy there
                w_new = learning_rule(w, X[N][:,k], Yc[N][j,k])
                queue.append((j, w_new))

        # save W results for each N
        W[N] = np.array([explored[i] for i in range(len(explored))])

    # return the loss values and weights
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

