import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.patches as mp
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True

solver = sys.argv[1]
N = int(sys.argv[2]) # works well with N=5

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

X = X.astype(int)
Y = Y.astype(int)
B = B.astype(bool)

def squares(ax, A, s, fc, ec, fs):
    for i, j in it.product(*map(range, A.shape)):
        # ax.add_patch(mp.Rectangle((j-.5, -i-.5), 1, 1, ec=ec[A[i,j]], fc=fc[A[i,j]], fill=True, zorder=int(A[i,j])))
        ax.add_patch(mp.Rectangle((j-.5, -i-.5), 1, 1, ec='k', fc=fc[A[i,j]], fill=True, zorder=int(A[i,j])))
        ax.text(j - .3, -i - .25, s[A[i,j]], fontsize=fs, color=ec[A[i,j]], zorder=int(A[i,j]))
    # ax.set_xlim([-.5, A.shape[1]-.5])
    # ax.set_ylim([.5, -A.shape[0]+.5])
    ax.axis("equal")
    ax.axis("off")

fig, axs = pt.subplots(3,1, sharex=True, height_ratios = (X.shape[1], Y.shape[0], B.shape[0]), figsize=(4,6))

squares(axs[0], X.T, s = {1: "$+$", -1: "$-$"}, fc={1: "w", -1: "k"}, ec={1: "k", -1: "w"}, fs=12)
axs[0].set_title("$X^T$", fontsize=14)

squares(axs[1], Y, s = {1: "$+$", -1: "$-$"}, fc={1: "w", -1: "k"}, ec={1: "k", -1: "w"}, fs=12)
axs[1].set_title("$Y$", fontsize=14)
axs[1].text(-1.5, -Y.shape[0]/2+.5, "$i$", fontsize=14)

squares(axs[2], B, s = {True: "T", False: "F"}, fc={True: "w", False: "k"}, ec={True: "k", False: "w"}, fs=10)
axs[2].set_title("$B$", fontsize=14)
axs[2].text(B.shape[1]//2-1, -B.shape[0]-1, "$k$", fontsize=14)

pt.tight_layout()
pt.savefig('enum_fig.pdf')
pt.show()

