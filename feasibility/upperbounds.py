import numpy as np
import matplotlib.pyplot as pt
import itertools as it

L = 2
Ns = np.arange(3, 32)
# Ns = np.arange(3, 20000)
Ms = []
for N in Ns:
    for M in it.count():
        if (M+1)*np.log2(M+1) > N*(N-1)**2 * L / 2: break
    Ms.append(M)
    
    # for logM in it.count():
    #     if 2**logM * logM > N*(N-1)**2 * L / 2: break
    # Ms.append(2**(logM-1))

pt.plot(Ns, Ns**1, 'k-', label="N")
# pt.plot(Ns, L*Ns**2, 'b-', label="L N2")
# pt.plot(Ns, L*Ns**2.5, 'g-', label="L N2.5")
pt.plot(Ns, Ms, 'r-', label="M")
pt.legend()
pt.xlabel("N")
pt.show()

