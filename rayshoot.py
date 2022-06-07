import numpy as np
import itertools as it

def rayshoot(u, v):
    N = len(u)
    crossings = {0: {(0,0): ()}}
    for i in range(N):
        crossings[i+1] = {}
        for (numer, denom), x in crossings[i].items():
            crossings[i+1][numer + u[i], denom + v[i]] = x + (+1,)
            crossings[i+1][numer - u[i], denom - v[i]] = x + (-1,)

    n_opt, d_opt, x_opt = None, None, None
    for (numer, denom), x in crossings[N].items():
        if denom == 0 or numer / denom < 0: continue
        if (n_opt == None) or (numer / denom < n_opt / d_opt):
            n_opt, d_opt = numer, denom
            x_opt = x

    return n_opt, d_opt, x_opt, crossings
    
if __name__ == "__main__":
    
    Ns = np.arange(2, 18)
    csizes = {}
    for N in Ns:
        print(f"{N}...")
        X = np.array(tuple(it.product((-1, +1), repeat=N)))

        csizes[N] = []
        for r in range(30):
    
            # u = np.random.randint(-2*N, 2*N+1, size=N)
            u = np.random.randint(-N, N+1, size=N)
            v = np.random.choice((-1, 1), size=N)
        
            n, d, x, c = rayshoot(u, v)
            csizes[N].append(sum(map(len, c.values())))
            
            # print(x, n, d)
            # if n != None: print(n/d)
        
            Xu, Xv = X @ u, X @ v
            alpha = Xu / Xv
            keep = (Xv != 0) & (alpha >= 0)

            if not keep.any():
                assert n == None

            else:

                ak = alpha[keep]
                mn = ak.min()
                Xm = X[keep][ak == mn]
                # print(mn)
                # print(Xm)
                # print((Xm == x).all(axis=1))

                assert (Xm == x).all(axis=1).any()
    
    import matplotlib.pyplot as pt
    pt.plot(Ns, 2**Ns, 'k--')
    for N, csz in csizes.items():
        pt.plot([N]*len(csz), csz, '.', color=(.5,)*3)
    pt.yscale("log")
    pt.show()
        
