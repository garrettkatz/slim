import numpy as np

def sylvester(K):
    # build sylvester matrix size 2**K, K >= 0
    H = np.array([[1]])
    for k in range(1, K+1):
        H = np.block([[H, H], [H, -H]])
    return H

# return sylvester-hadamard vector with largest dot product with x
def cleanup(x):
    N = len(x)
    K = int(np.log2(N).round())

    # split into cases
    x = x.copy()
    for k in range(1,K+1):
        x = x.reshape((2**k, -1))
        x[ ::2] = x[::2] +   x[1::2]
        x[1::2] = x[::2] - 2*x[1::2] # -2 for in-place

    # reconstruct h
    idx = np.argmax(x.reshape(N))
    h = np.empty(N, dtype=int) # pre-allocate
    h[0] = 1
    for k in range(K):
        sign = (-1)**(idx % 2)
        idx = idx // 2
        h[2**k:2**(k+1)] = sign * h[:2**k]

    return h

def generalized_sylvester(K):
    H = np.array([[1]])
    ts = []
    for k in range(1, K+1):
        # t = np.random.rand()*np.pi/2
        # t = np.linspace(0, np.pi, K+2)[-k-1]
        t = np.linspace(0, np.pi, K+2)[k]
        c, s = np.cos(t), np.sin(t)
        H = np.block([[c*H, s*H], [s*H, -c*H]])
        ts.append(t)
        # print(f"gen syl {k=}: {t=:.3f}")

    assert np.allclose(H.T @ H, np.eye(len(H)))

    return H, ts

# return sylvester-hadamard vector with largest dot product with x
def generalized_cleanup(x, ts):
    N = len(x)
    K = int(np.log2(N).round())

    # split into cases
    x = x.copy()
    for k, t in zip(range(1,K+1), reversed(ts)):
        c, s = np.cos(t), np.sin(t)
        x = x.reshape((2**k, -1))
        xp = c*x[::2] + s*x[1::2]
        xn = s*x[::2] - c*x[1::2]
        x[ ::2] = xp
        x[1::2] = xn

    # reconstruct h
    idx = np.argmax(x.reshape(N))
    h = np.empty(N) # pre-allocate
    h[0] = 1.
    for k, t in zip(range(K), ts):
        sign = (-1)**(idx % 2)
        idx = idx // 2
        a, b = np.cos(t), np.sin(t)
        if sign < 0: a, b = b, -a
        h[2**k:2**(k+1)] = b * h[:2**k]
        h[    :2**k    ] = a * h[:2**k]

    return h

if __name__ == "__main__":

    import pickle as pk
    import matplotlib.pyplot as pt
    from matplotlib import rcParams

    # config for big check
    do_check = False
    num_reps = 10
    K_max = 13

    if do_check:
        # check a few sylvester hadamards
        print("sanity checks...")
        for K in range(8):
            N = 2**K
            H = sylvester(K)
            Hg, ts = generalized_sylvester(K)
            # print(K, N)
            # print(H)
    
            # make sure orthogonal
            assert np.allclose(N * np.eye(N), H.T @ H)
            assert np.allclose(np.eye(N), Hg.T @ Hg)
    
            # make sure vectors clean-up to themselves
            for x in H:
                h = cleanup(x)
                assert (x == h).all()
            for x in Hg:
                hg = generalized_cleanup(x, ts)
                assert np.allclose(x, hg)
    
    
        # big test for correctness and timing
        from time import perf_counter

        runtimes = {"slow": {}, "fast": {}, "slog": {}, "gend": {}}
    
        for K in range(1,K_max+1):
            print(f"timing {K}...")
    
            H = sylvester(K)
            Hg, ts = generalized_sylvester(K)
            N = 2**K
    
            runtimes["slow"][K] = []
            runtimes["fast"][K] = []
            runtimes["slog"][K] = []
            runtimes["gend"][K] = []
    
            for rep in range(num_reps):
    
                x = np.random.randn(N)
    
                start = perf_counter()
                h_slow = H[(H @ x).argmax()] # H is symmetric
                duration = perf_counter() - start
                runtimes["slow"][K].append(duration)
    
                start = perf_counter()
                h_fast = cleanup(x)
                duration = perf_counter() - start
                runtimes["fast"][K].append(duration)

                start = perf_counter()
                h_slog = Hg[(Hg.T @ x).argmax()]
                duration = perf_counter() - start
                runtimes["slog"][K].append(duration)

                start = perf_counter()
                h_gend = generalized_cleanup(x, ts)
                duration = perf_counter() - start
                runtimes["gend"][K].append(duration)
                
                assert (h_slow == h_fast).all()
                assert (h_slog == h_gend).all()
    
        with open("hadamard_cleanup.pkl","wb") as f: pk.dump(runtimes, f)

    with open("hadamard_cleanup.pkl","rb") as f: runtimes = pk.load(f)

    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 12
    rcParams["text.usetex"] = True

    pt.figure(figsize=(5,3))
    markers = {"fast": "+", "slow": "o", "gend": "x", "slog": "v"}
    for key, data in runtimes.items():
        for i, (K, results) in enumerate(data.items()):
            if i == 0:
                pt.plot(K + 0.05*np.random.randn(len(results)), results, markers[key], mec="k", mfc="none", linestyle="none", label=key)
            else:
                pt.plot(K + 0.05*np.random.randn(len(results)), results, markers[key], mec="k", mfc="none", linestyle="none")

    pt.legend()
    pt.xlabel("$\\mathrm{log}_2 N$")
    pt.ylabel("Runtime (seconds)")
    pt.yscale("log")
    pt.tight_layout()
    pt.savefig("hadamard_cleanup.eps")
    pt.show()

