import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import rcParams
from lam import hrr_conv, hrr_read
from hadamard_cleanup import sylvester, generalized_sylvester

# print(sylvester(3))
do_run = False
K_min, K_max = 4, 12
Ms = (1, 2, 10, 20)
num_reps = 30
kinds = ("Gaussian", "Binary", "Sylvester", "Generalized")

if do_run:

    recalls = {}
    for M in Ms:
        recalls[M] = {}
    
        for K in range(K_min, K_max+1):
            recalls[M][K] = {kind: [] for kind in kinds}

            N = 2**K
            
            for rep in range(100):
    
                # value representations
                V = {
                    "Gaussian": np.random.randn(N,N) / N**.5, # standard gaussian
                    "Binary": np.sign(np.random.randn(N,N)) / N**.5, # standard discrete
                    "Sylvester": sylvester(K) / N**.5, # hadamard
                    "Generalized": generalized_sylvester(K)[0], # hadamard
                }

                # address representations
                A = np.random.randn(M, N) / N**.5
            
                for kind in kinds:
        
                    idx = np.random.randint(len(V[kind]), size=M)
        
                    # write memory
                    mem = np.zeros(N)
                    for i in range(M):
                        v = V[kind][idx[i]]
                        mem += hrr_conv(A[i], v)
        
                    # read mem
                    idx_clean = np.empty(M, dtype=int)
                    for i in range(M):
                        u_noisy = hrr_read(mem, A[i])
                        if kind == "Binary":
                            u_clean = np.sign(u_noisy) / N**.5
                            matches = np.allclose(V[kind] == u_clean, axis=-1)
                            if matches.any():
                                idx_clean[i] = matches.argmax()
                            else:
                                idx_clean[i] = N # will never match idx[i]
                        else:
                            idx_clean[i] = (V[kind] @ u_noisy).argmax()
        
                    # performance
                    recalls[M][K][kind].append((idx == idx_clean).all())
            
            # results = np.concatenate(results)
            # print(f"{np.mean(results)} +/- {np.std(results)} vs {1}")
            for kind in kinds:
                print(f"  recall rate {M} {K} {kind}: {np.mean(recalls[M][K][kind])}")

    with open("hadamard_noise.pkl","wb") as f: pk.dump(recalls, f)

with open("hadamard_noise.pkl","rb") as f: recalls = pk.load(f)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12
markers = {
    "Gaussian": "+",
    "Binary": "x",
    "Sylvester": "s",
    "Generalized": "o",
}
fig, axs = pt.subplots(1, len(Ms), figsize=(10,3), constrained_layout=True)
for m, M in enumerate(Ms):
    for kind in kinds:
        x = 2**np.arange(K_min, K_max+1)
        y = [np.mean(recalls[M][K][kind]) for K in range(K_min, K_max+1)]
        axs[m].plot(x, y, color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)
    axs[m].set_title(f"$M = {M}$")
    if m == 0: axs[m].set_ylabel("Success Rate")
    # pt.xlabel("Vector dimension")
    axs[m].set_xscale("log", base=2)
    axs[m].set_ylim([-.1, 1.1])
    if m == 0: axs[m].legend()
fig.supxlabel("$N$")
# pt.tight_layout()
pt.savefig("sylvester_success.eps")
pt.show()

pt.figure(figsize=(5,3))
for kind in kinds:
    Ks = []
    for M in Ms:
        reliable = np.flatnonzero([np.all(recalls[M][K][kind]) for K in range(K_min, K_max+1)])
        K_M = K_min + reliable.min() if len(reliable) > 0 else np.inf
        Ks.append(K_M)
    pt.plot(Ms, 2**np.array(Ks), color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)
pt.xlabel("$M$")
pt.ylabel("$N^*_M$", rotation=0)
pt.yscale("log",base=2)
pt.legend()
pt.tight_layout()
pt.savefig("sylvester_capacity.eps")
pt.show()


