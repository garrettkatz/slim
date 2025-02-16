import itertools as it
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import rcParams
from lam import hrr_conv, hrr_read
from hadamard_cleanup import generalized_sylvester, generalized_cleanup

def binary_cleanup(u):
    return np.sign(u) / N**.5

# # good example settings
# M = 10
# K = 9

do_run = False
Ms = (1, 2, 10)
Ks = tuple(range(4, 11))
T = 30
num_reps = 10

if do_run:
    success_rate = {}
    for (M, K, rep) in it.product(Ms, Ks, range(num_reps)):
        print(f"{M=}, {K=}, {rep=}")

        N = 2**K        
        A = np.random.randn(M,N) / N**.5
        V_gen, thetas = generalized_sylvester(K)
        V_bin = np.random.choice([-1, 1], size=(N,N)) / N**.5
        V_hrr = np.random.randn(N,N) / N**.5
        
        ref_memory = np.zeros(M, dtype=int)
        gen_memory = np.zeros(N)
        bin_memory = np.zeros(N)
        hrr_memory = np.zeros(N)
        for m in range(M):
            gen_memory += hrr_conv(A[m], V_gen[0])
            bin_memory += hrr_conv(A[m], V_bin[0])
            hrr_memory += hrr_conv(A[m], V_hrr[0])
        
        success_rate[M,K,rep] = {key: [] for key in ("gen","bin","hrr")}
        for t in range(T):
            m = np.random.randint(M)
            i = np.random.randint(N)
            ref_memory[m] = i
        
            u = hrr_read(gen_memory, A[m])
            v = generalized_cleanup(u, thetas)
            gen_memory += hrr_conv(A[m], V_gen[i]) - hrr_conv(A[m], v)
        
            u = hrr_read(bin_memory, A[m])
            v = binary_cleanup(u)
            # v = V_bin[(V_bin @ u).argmax()]
            bin_memory += hrr_conv(A[m], V_bin[i]) - hrr_conv(A[m], v)
        
            v = hrr_read(hrr_memory, A[m])
            hrr_memory += hrr_conv(A[m], V_hrr[i]) - hrr_conv(A[m], v)
        
            success = []
            for m, i in enumerate(ref_memory):
                u = hrr_read(gen_memory, A[m])
                v = generalized_cleanup(u, thetas)
                success.append( np.allclose(V_gen[i], v) )
            success_rate[M,K,rep]["gen"].append(np.mean(success))
        
            success = []
            for m, i in enumerate(ref_memory):
                u = hrr_read(bin_memory, A[m])
                v = V_bin[(V_bin @ u).argmax()]
                success.append( np.allclose(V_bin[i], v) )
            success_rate[M,K,rep]["bin"].append(np.mean(success))
        
            success = []
            for m, i in enumerate(ref_memory):
                u = hrr_read(bin_memory, A[m])
                j = (V_hrr @ u).argmax()
                success.append( i == j )
            success_rate[M,K,rep]["hrr"].append(np.mean(success))

    with open("sylvester_mutable.pkl","wb") as f: pk.dump(success_rate, f)

with open("sylvester_mutable.pkl","rb") as f: success_rate = pk.load(f)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12

markers = {"gen": "o", "bin": "+", "hrr": "x"}
M, K = 10, 9
rates = {
    key: np.array([success_rate[M,K,rep][key] for rep in range(num_reps)])
    for key in markers.keys()}
pt.figure(figsize=(5,3))
for key in markers.keys():
    pt.plot(rates[key].mean(axis=0), 'k-' + markers[key], mfc="none", label=key)
pt.xlabel("Time-step")
pt.ylabel("Retrieval Rate")
pt.legend()
pt.tight_layout()
pt.savefig("sylvester_mutable_sample.eps")
pt.show()

fig, axs = pt.subplots(1, len(Ms), figsize=(10,3), constrained_layout=True)
for m, M in enumerate(Ms):
    rates = {
        key: np.array([
            np.mean([success_rate[M,K,rep][key] for rep in range(num_reps)])
            for K in Ks])
        for key in markers.keys()}

    for key in markers.keys():
        axs[m].plot(2**np.array(Ks), rates[key], 'k-' + markers[key], mfc="none", label=key)
    axs[m].set_xscale("log", base=2)
    axs[m].set_ylim([-.1,1.1])
    axs[m].set_title(f"$M = {M}$")
    if m == 0:
        axs[m].set_ylabel("Time-Averaged Retrieval Rate")
        axs[m].legend()
fig.supxlabel("$N$")
pt.savefig("sylvester_mutable.eps")
pt.show()

