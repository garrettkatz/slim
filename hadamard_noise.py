import numpy as np
import matplotlib.pyplot as pt
from lam import hrr_conv, hrr_read
from hadamard_cleanup import sylvester, cleanup

# print(sylvester(3))

H = sylvester(7)
N = len(H)
print(f"N={N}")

results = []
recalls = []
for rep in range(100):
    c = np.random.randn(N) / N**.5
    h = H[np.random.randint(N)]
    # h = np.random.randn(N) / N**.5
    h_noisy = hrr_read(hrr_conv(c, h), c)
    h_clean = cleanup(h_noisy)
    # results.append((h_noisy - h)[np.random.randint(N)])
    results.append(h_noisy - h)
    recalls.append(int((h == h_clean).all()))
results = np.concatenate(results)
print(f"{np.mean(results)} +/- {np.std(results)} vs {1}")
print(f"recall rate = {np.mean(recalls)}")

pt.hist(results)
pt.show()


