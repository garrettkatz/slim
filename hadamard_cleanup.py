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
        # t = np.random.rand()*2*np.pi
        t = np.linspace(0, 2*np.pi, K+2)[k]
        # t = np.linspace(0, 2*np.pi, K)[k-1]
        c, s = np.cos(t), np.sin(t)
        H = np.block([[c*H, s*H], [s*H, -c*H]])
        ts.append(t)
        # print(f"gen syl {k=}: {t=:.3f}")

    assert np.allclose(H.T @ H, np.eye(len(H)))

    return H, ts

def generalized_reconstruct(thetas, idx):
    v = np.array([1.])
    for k, t in enumerate(thetas):
        if (idx >> k) & 1 == 0:
            v = np.concatenate([np.cos(t) * v,  np.sin(t) * v])
        else:
            v = np.concatenate([np.sin(t) * v, -np.cos(t) * v])
    return v

# return sylvester-hadamard vector with largest dot product with x
def generalized_cleanup(u, thetas):
    N = len(u)
    K = int(np.log2(N).round())

    # split into cases
    u = u.copy()
    for k, t in zip(range(1,K+1), reversed(thetas)):
        c, s = np.cos(t), np.sin(t)
        u = u.reshape((2**k, -1))
        u[::2], u[1::2] = c*u[::2] + s*u[1::2], s*u[::2] - c*u[1::2]

    # reconstruct v*
    idx = np.argmax(u.reshape(N))
    v = generalized_reconstruct(thetas, idx)
    # v = np.array([1.])
    # for k, t in zip(range(K), thetas):
    #     a, b = np.cos(t), np.sin(t)
    #     if idx & 1 == 1: a, b = b, -a
    #     idx = idx >> 1
    #     v = np.concatenate([a * v, b * v])

    return v

if __name__ == "__main__":

    import pickle as pk
    import matplotlib.pyplot as pt
    from matplotlib import rcParams

    # config for big check
    do_check = True
    num_reps = 30
    K_max = 14

    if do_check:
        # check a few sylvester hadamards
        print("sanity checks...")
        for K in range(8):
            N = 2**K
            V, thetas = generalized_sylvester(K)
            # H = sylvester(K)
            # print(K, N)
            # print(H)
    
            # make sure orthogonal
            # assert np.allclose(N * np.eye(N), H.T @ H)
            assert np.allclose(np.eye(N), V.T @ V)
    
            # make sure vectors clean-up to themselves
            # for x in H:
            #     h = cleanup(x)
            #     assert (x == h).all()
            for v in V:
                u = generalized_cleanup(v, thetas)
                assert np.allclose(u, v)

        # big test for correctness and timing
        from time import perf_counter

        runtimes = {"Brute-Force": {}, "Efficient": {}}
    
        for K in range(1,K_max+1):
            print(f"timing {K}...")
            N = 2**K
            V, thetas = generalized_sylvester(K)

            runtimes["Brute-Force"][K] = []
            runtimes["Efficient"][K] = []
    
            for rep in range(num_reps):

                # V, thetas = generalized_sylvester(K)
                x = np.random.randn(N)
    
                # start = perf_counter()
                # h_slow = H[(H @ x).argmax()] # H is symmetric
                # duration = perf_counter() - start
                # runtimes["slow"][K].append(duration)
    
                # start = perf_counter()
                # h_fast = cleanup(x)
                # duration = perf_counter() - start
                # runtimes["fast"][K].append(duration)

                start = perf_counter()
                u_brute = V[(V.T @ x).argmax()]
                duration = perf_counter() - start
                runtimes["Brute-Force"][K].append(duration)

                start = perf_counter()
                u_eff = generalized_cleanup(x, thetas)
                duration = perf_counter() - start
                runtimes["Efficient"][K].append(duration)
                
                assert np.allclose(u_brute, u_eff)
    
        with open("hadamard_cleanup.pkl","wb") as f: pk.dump(runtimes, f)

    with open("hadamard_cleanup.pkl","rb") as f: runtimes = pk.load(f)

    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 12
    rcParams["text.usetex"] = True

    pt.figure(figsize=(10,3))
    markers = {"Efficient": "+", "Brute-Force": "."}
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
    pt.savefig("cleanup_timing.eps")
    pt.show()

