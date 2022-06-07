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

    N = 5
    X = np.array(tuple(it.product((-1, +1), repeat=N)))

    u = np.random.randint(-3, 4, size=N)
    v = np.random.choice((-1, 1), size=N)

    n, d, x, c = rayshoot(u, v)
    print(x)
    print(n, d, n/d)

    Xu, Xv = X @ u, X @ v
    alpha = Xu / Xv
    keep = (Xv != 0) & (alpha >= 0)
    mn = alpha[keep].min()
    print(mn)
    print(X[keep][alpha[keep] == mn])

    print((X[keep][alpha[keep] == mn] == x).all(axis=1))
    
    

