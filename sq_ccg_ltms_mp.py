import sys
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mp
from numpy.linalg import norm
from numpy.polynomial import Polynomial
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count

from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)

mp.rcParams['font.family'] = 'serif'

# save one core when multiprocessing
num_procs = cpu_count()-1

# helper for pooled descent direction linear programs
def descent_direction(args):
    c, A_ub, b_ub, A_eq, b_eq = args

    # run the linear program
    result = linprog(*args,
        bounds = (None, None),
        method = 'simplex', # other methods may miss some solutions
    )

    u_r = result.x
    return u_r

# @profile
def main():

    do_opt = True # whether to run the optimization or just load results
    num_updates = int(1e6) # maximum number of updates if stopping criterion not reached
    save_period = int(1e3) # number of updates between saving intermediate results
    eps = 0.1 # constraint slack threshold

    # input dimension for optimization
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 4

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)
    with open(f"adjs_{N}_jc.npz", "rb") as f:
        Ac = pk.load(f)

    # set up boundary indices to remove redundant region constraints
    K = {}
    for i in Yn:
        K[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    # batch set up of projection matrices, PX[k] is matrix for vertex k
    PX = np.eye(N) - X.T.reshape(-1, N, 1) * X.T.reshape(-1, 1, N) / N

    # save original lp weights
    W_lp = Wc.copy()

    # # set up equality constraints for weights invariant to same symmetries as their regions
    # # doesn't converge to zero span loss
    # A_sym = []
    # for i, w in enumerate(Wc):
    #     w = w.round().astype(int) # will not work past n=8
    #     A_sym_i = np.eye(N-1,N) - np.eye(N-1,N,k=1) # retain w[n] - w[n+1] == 0
    #     A_sym_i = A_sym_i[w[:-1] == w[1:]] # i.e. w[n] == w[n+1] as constraint
    #     if w[0] == 0: # zeros must stay so
    #         A_sym_i = np.concatenate((np.eye(1,N), A_sym_i), axis=0)
    #     A_sym.append(A_sym_i)

    if do_opt:

        # set up constant descent direction args once
        dd_args = []
        for r in range(len(W_lp)):

            # counteract weight norm shrinking
            A_eq = W_lp[r:r+1].copy()
            b_eq = (A_eq**2).sum(axis=1)

            # irredundant region constraints (only boundaries)
            A_ub = -(X[:,K[r]] * Yc[r, K[r]]).T
            b_ub = -np.ones(len(K[r]))*eps

            # save for later
            dd_args.append((A_ub, b_ub, A_eq, b_eq))

        # optimization metrics
        loss_curve = [] # span loss
        grad_curve = [] # projected gradient norm
        slack_curve = [] # constraint slack
        angle_curve = [] # max angle
        gamma_curve = [] # line search step size

        # optimization loop
        for update in range(num_updates):

            # calculate loss and gradient on joint-canonical adjacencies
            loss = 0.
            min_cos = 1.
            grad = np.zeros(Wc.shape)
            for (i,j,k) in Ac:

                # get projected weights
                wi, wj, xk = Wc[i], Wc[j], X[:,k]
                Pk = PX[k]
                wiPk = wi @ Pk
                wjPk = wj @ Pk
                wiPkwi = wiPk @ wiPk
                wiPkwj = wiPk @ wjPk
                wjPkwj = wjPk @ wjPk
                wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

                # track minimum cosine
                min_cos = min(min_cos, wiPkwj / (wiPk_n * wjPk_n))
    
                # accumulate span loss
                loss += wiPkwi * wjPkwj - wiPkwj**2
    
                # accumulate gradient
                grad[i] += 4 * (wiPk * wjPkwj - wiPkwj * wjPk)

            max_angle = np.arccos(min_cos)
            angle_curve.append(max_angle)
            loss_curve.append(loss)

            # Pooled decent direction calculation
            args = [(grad[r].copy(),) + args for r,args in enumerate(dd_args)]
            with Pool(num_procs) as pool:
                u = pool.map(descent_direction, args)
            U = np.vstack(u)
            Delta = U - Wc

            # analytical line search
            cubic = [0, 0, 0, 0] # coefficients of line search derivative (cubic polynomial)
            step_grad = np.zeros(grad.shape) # sanity check gradient at end of step (gamma = 1)
            for (i,j,k) in Ac:

                # get weight and delta projections
                di, dj, wi, wj, xk = Delta[i], Delta[j], Wc[i], Wc[j], X[:,k]
                Pk = PX[k]
                wiPk = wi @ Pk
                wjPk = wj @ Pk
                diPk = di @ Pk
                djPk = dj @ Pk

                # accumulate span loss gradient at gamma = 1
                vi, vj = wiPk + diPk, wjPk + djPk
                step_grad[i] += 4 * (vi * (vj @ vj) - (vi @ vj) * vj)

                # accumulate cubic coefficients
                cubic[0] += wjPk @ (wjPk[:,np.newaxis] * wiPk - wiPk[:,np.newaxis] * wjPk) @ diPk
                cubic[1] += 2*(wjPk @ djPk)*(wiPk @ diPk) + (wjPk @ wjPk)*(diPk @ diPk) \
                            - (wiPk @ wjPk)*(djPk @ diPk) - (wjPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
                cubic[2] += 2*(wjPk @ djPk)*(diPk @ diPk) + (wiPk @ diPk)*(djPk @ djPk) \
                            - (wjPk @ diPk)*(diPk @ djPk) - (djPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
                cubic[3] += djPk @ (djPk[:,np.newaxis] * diPk - diPk[:,np.newaxis] * djPk) @ diPk

            cubic = [4*cub for cub in cubic]

            # sanity check correct cubic coefficients at gamma = 0 and gamma = 1
            assert np.allclose(cubic[0], (grad * Delta).sum())
            assert np.allclose(sum(cubic), (step_grad * Delta).sum())

            # get extrema of line search objective in [0, 1]
            cubic = Polynomial(cubic)
            roots = cubic.roots()
            gammas = np.append(roots.real, (0., 1.))
            gammas = gammas[(0 <= gammas) & (gammas <= 1)]

            # evaluate extrema to find global min
            quartic = cubic.integ()
            evals = gammas[:,np.newaxis]**np.arange(5) @ quartic.coef
            gamma = gammas[evals.argmin()]

            # take step
            gamma_curve.append(gamma)
            Wc = Wc + gamma * Delta # stays in interior as long as 0 <= gamma <= 1

            # calculate norm of projected gradient
            pgnorm = np.fabs((grad * Delta).sum()) / norm(Delta)
            grad_curve.append(pgnorm)

            # stop if infeasible, should not happen (but numerical issues when eps = 0)
            if eps > 0:
                feasible = (np.sign(Wc @ X) == Yc).all()
                if not feasible:
                    print("infeasible")
                    break
    
            # constraint slack
            slack = np.fabs(Wc @ X).min()
            slack_curve.append(slack)

            message = f"{update}/{num_updates}: loss={loss}, angle<={max_angle}), |pgrad|={pgnorm**0.5}, slack={slack}, gamma={gamma}"
            print(message)

            # check stopping criterion
            early_stop = (gamma == 0.)

            # save intermediate or final results
            if early_stop or (update + 1 == num_updates) or (update % save_period == 0):
                with open(f"sq_ccg_ltm_mp_{N}.pkl", "wb") as f:
                    pk.dump((Wc, loss_curve, slack_curve, grad_curve, angle_curve, gamma_curve), f)

            if early_stop:
                print("stopping early, reached gamma=0")
                break

    # load results
    with open(f"sq_ccg_ltm_mp_{N}.pkl", "rb") as f:
        (Wc, loss_curve, slack_curve, grad_curve, angle_curve, gamma_curve) = pk.load(f)

    # suppress large wall of text for N >= 6
    if N < 6:
        np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})

        print("\n" + "*"*8 + " change in weights " + "*"*8 + "\n")
    
        for i in range(len(Wc)):
            print(W_lp[i], Wc[i], np.fabs(Wc[i] @ X).min())
    
        print("\n" + "*"*8 + " span coefficients and residuals " + "*"*8 + "\n")
        print("wi ~ a * wj + b * xk, resid, ijk")
        for (i, j, k) in Ac:
            ab = np.linalg.lstsq(np.vstack((Wc[j], X[:,k])).T, Wc[i], rcond=None)[0]
            resid = np.fabs(Wc[i] - (ab[0]*Wc[j] + ab[1]*X[:,k])).max()
            print(Wc[i], ab[0], Wc[j], ab[1], X[:,k], resid, i,j,k)

if __name__ == "__main__": main()


