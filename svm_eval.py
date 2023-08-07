import numpy as np
from sklearn.svm import LinearSVC

"""
Evaluates a learning rule on randomly sampled training runs
Each run is a random subset of examples from one dichotomy in a given dimension

Inputs:
    X[N]: half-cube for dimension N
    Y[N]: dichotomies from enumerate_ltms for dimension N
    update_rule(w, x, y, N): returns new weights w'
    num_runs: number of training runs for evaluation
Outputs: (loss, accu) arrays with one entry per run
    loss[run]: angle between learned and max-margin weights
    accu[run]: training accuracy of the learned weights
"""
def svm_eval(X, Y, update_rule, num_runs):

    accu = np.empty(num_runs)
    loss = np.empty(num_runs)
    Ns = np.array(list(Y.keys()))
    for run in range(num_runs):
        
        # randomly sample training data
        N = np.random.choice(Ns) # dimension
        i = np.random.choice(Y[N].shape[0]) # dichotomy
        T = np.random.choice(Y[N].shape[1]) + 1 # number of examples
        K = np.random.choice(Y[N].shape[1], size=T, replace=False) # example indices

        # get max-margin classifier via svm
        svc = LinearSVC(dual='auto', fit_intercept=False)
        svc.fit(
            np.concatenate((X[N][:,K], -X[N][:,K]), axis=1).T,
            np.concatenate((Y[N][i,K], -Y[N][i,K]), axis=0))

        # apply learning rule to examples
        w = np.zeros(N)
        for t, k in enumerate(K):
            w = update_rule(w, X[N][:, k], Y[N][i, k], N)

        # compare angle between max-margin weights and learned weights
        if np.linalg.norm(w) > 0:
            cos = (svc.coef_ @ w) / (np.linalg.norm(svc.coef_) * np.linalg.norm(w))
            loss[run] = np.arccos(np.clip(cos, -1, +1)) # avoid NaNs from small round-off error
        else:
            # zero weight vector considered maximally distant from max margin
            loss[run] = np.pi

        # save accuracy
        accu[run] = (np.sign(w @ X[N][:,K]) == Y[N][i,K]).mean()

    return loss, accu

if __name__ == "__main__":

    # define function for perceptron learning rule
    def perceptron_rule(w, x, y, N):
        return w + (y - np.sign(w @ x)) * x

    # load halfcubes and linearly separable dichotomies
    X, Y = {}, {}
    for N in range(3, 6):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    # evaluate perceptron compared to SVM
    loss, accu = svm_eval(X, Y, perceptron_rule, num_runs=200)
    loss = loss * 180/np.pi # convert to degrees

    import matplotlib.pyplot as pt

    pt.subplot(1,2,1)
    pt.hist(loss)
    pt.ylabel("Frequency")
    pt.xlabel(f"Angle (deg) ~ {loss.mean():.3f}")

    pt.subplot(1,2,2)
    pt.hist(accu)
    pt.xlabel(f"Accuracy ~ {accu.mean():.3f}")

    pt.show()

