import os
import numpy as np
import pickle
import torch

from make_data import DataSet

data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_data")

def train():
    N = 4
    ltms_file = os.path.join(data_root,"ltms_{}_c.npz".format(N))
    sq_ccg_file = os.path.join(data_root, "sq_ccg_ltm_mp_{}.pkl".format(N))
    adj_file = os.path.join(data_root, "adjs_{}_jc.npz".format(N))

    ltms = np.load(ltms_file)
    Y, X = ltms["Y"], ltms["X"]
    with open(sq_ccg_file, "rb") as f:
        Wc = pickle.load(f)[0]
    with open(adj_file, "rb") as f:
        Ac = pickle.load(f)
    # each iteration of this loop is a row in Table 1
    # x_stack = None
    # w_prev_stack = None
    # y_stack = None
    # w_next_stack = None
    # for (i, j, k) in Ac:
    #     x = X[:, k].astype(np.float64)
    #     y = np.array([Y[i, k]]).astype(np.float64)
    #     w_prev = Wc[j].astype(np.float64)
    #     w_next = Wc[i].astype(np.float64)
    #     x_stack = x if x_stack is None else np.vstack((x_stack, x))
    #     y_stack = y if y_stack is None else np.vstack((y_stack, y))
    #     w_prev_stack = w_prev if w_prev_stack is None else np.vstack((w_prev_stack, w_prev))
    #     w_next_stack = w_next if w_next_stack is None else np.vstack((w_next_stack, w_next))
    # # create dataset
    # dataset = DataSet(x_stack, y_stack, w_prev_stack, w_next_stack)
    input_stack = None
    for (i, j, k) in Ac:
        concat_val = np.concatenate((X[:, k].astype(float), 
                                     Wc[j].astype(float), 
                                     Y[i,k].astype(float)))
        input_stack = concat_val if input_stack is None else np.vstack((input_stack, concat_val))
    dataset = DataSet(input_stack)

    

if __name__ == "__main__":
    train()