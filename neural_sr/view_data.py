import numpy as np
import os

if __name__ == "__main__":
    fname  = "/home/emilioliu/learning/research/one_shot_full_capacity/katz_codes/slim/ltms_{}_c.npz"
    ltms = np.load(fname.format(4))
    Y, X = ltms["Y"], ltms["X"]

