import numpy as np
import pickle

def show_data(N):
    ltms = np.load("/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/saved_data/ltms_{}_c.npz".format(N))
    Y, X = ltms["Y"], ltms["X"]
    # dir_name = "/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/util"
    with open(f"/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/saved_data/sq_ccg_ltm_mp_{N}.pkl", "rb") as f:
        Wc = pickle.load(f)[0]
    with open(f"/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/saved_data/adjs_{N}_jc.npz", "rb") as f:
        Ac = pickle.load(f)
    # each iteration of this loop is a row in Table 1
    for (i, j, k) in Ac:
        # a, b are alpha and beta
        # these are the output for the formulas
        a, b = np.linalg.lstsq(np.vstack((Wc[j], X[:,k])).T,Wc[i],rcond=None)[0]
        print("alpha: {}, beta: {}".format(a, b))

        # inputs for the formulas are:
        # X[:,k]: the current training example input
        # Y[i,k]: the current training example label
        # Wc[j]: the previous weight vector
        # N: the dimensionality of the vectors


if __name__ == "__main__":
    N = 4
    fname  = "/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/saved_data/ltms_{}_c.npz"
    ltms = np.load(fname.format(N))
    Y, X = ltms["Y"], ltms["X"]
    print("X:{}".format(X.shape))
    print(X)
    print("Y:{}".format(Y.shape))
    print(Y)

    # show_data(N)
