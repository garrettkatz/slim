import numpy as np
import pickle as pk

### load data
def load_data(Ns):
    dataset = []
    for N in Ns:

        # canonical dichotomies
        ltms = np.load(f"ltms_{N}_c.npz") # _c for canonical
        Y, X = ltms["Y"], ltms["X"]

        # adjacencies between them
        with open(f"adjs_{N}_jc.npz", "rb") as f:
            Ac = pk.load(f)

        # span-loss-optimized weights
        with open(f"sq_ccg_ltm_mp_{N}.pkl", "rb") as f:
            Wc = pk.load(f)[0]
    
        # each iteration of this loop is a row in Table 1
        w_j, x_k, y_ik, alpha, beta = [], [], [], [], []
        for (i, j, k) in Ac:
            # a, b are alpha and beta
            # these are the output for the formulas
            a, b = np.linalg.lstsq(
                np.vstack((Wc[j], X[:,k])).T,
                Wc[i],
                rcond=None)[0]
            alpha.append(a)
            beta.append(b)
    
            # inputs for the formulas are:
            # Wc[j]: the previous weight vector
            # X[:,k]: the current training example input
            # Y[i,k]: the current training example label
            w_j.append(Wc[j])
            x_k.append(X[:,k])
            y_ik.append(Y[i,k])
            # N is recovered by the Dimension operator below

        # collect into arrays
        w_j = np.stack(w_j)
        x_k = np.stack(x_k)
        y_ik = np.array(y_ik).reshape(-1,1) # column vector
        alpha = np.array(alpha).reshape(-1,1) # column vector
        beta = np.array(beta).reshape(-1,1) # column vector

        # each N is one "line" of the geneng dataset
        dataset.append([w_j, x_k, y_ik, alpha, beta])

    return dataset        

if __name__ == "__main__":
    dataset = load_data([4,5])
    print(dataset)

