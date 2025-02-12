import numpy as np
import matplotlib.pyplot as pt
from lam import hrr_conv, hrr_read
from hadamard_cleanup import sylvester, cleanup

# print(sylvester(3))
K_min, K_max = 4, 12
num_in = 10

recalls = {}
for K in range(K_min, K_max+1):

    N = 2**K
    print(f"N={N}")
    
    # codecs
    H = {
        "g": np.random.randn(N,N) / N**.5, # standard gaussian
        "b": np.sign(np.random.randn(N,N)) / N**.5, # standard discrete
        "h": sylvester(K), # hadamard
    }
    
    recalls[K] = {key: [] for key in H.keys()}
    
    for rep in range(1000):
        c = np.random.randn(num_in, N) / N**.5
    
        for key in H.keys():

            idx = np.random.randint(len(H[key]), size=num_in)

            # write mem
            mem = np.zeros(N)
            for i in range(num_in):
                h = H[key][idx[i]]
                mem += hrr_conv(c[i], h)

            # read mem
            idx_clean = np.empty(num_in, dtype=int)
            for i in range(num_in):
                h_noisy = hrr_read(mem, c[i])
                idx_clean[i] = (H[key] @ h_noisy).argmax()

            # performance
            recalls[K][key].append((idx == idx_clean).all())
    
    # results = np.concatenate(results)
    # print(f"{np.mean(results)} +/- {np.std(results)} vs {1}")
    for key in H.keys():
        print(f"  recall rate {key}: {np.mean(recalls[K][key])}")

for label in ("gaussian","binary","hadamard"):
    key = label[0]
    x = 2**np.arange(K_min, K_max+1)
    y = [np.mean(recalls[K][key]) for K in range(K_min, K_max+1)]
    pt.plot(x, y, label=label)

pt.xlabel("Vector dimension")
pt.xscale("log", base=2)
pt.ylabel(f"{num_in} pairs recall rate")
pt.legend()
pt.tight_layout()
pt.show()
# pt.hist(results)
# pt.show()


