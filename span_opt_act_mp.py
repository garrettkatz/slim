import sys
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mp
from numpy.linalg import norm
from numpy.polynomial import Polynomial
import scipy.sparse as sp
import scipy.linalg as sl
from multiprocessing import Pool, cpu_count

from scipy.optimize import linprog, OptimizeWarning
import warnings
warnings.filterwarnings("ignore", category=sl.LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)

mp.rcParams['font.family'] = 'serif'

# @profile
def main():

    do_opt = True # whether to run the optimization or just load results
    num_updates = int(1e7) # maximum number of updates if stopping criterion not reached
    save_period = int(1e4) # number of updates between saving intermediate results
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

    # set up active constraint flags
    active = np.zeros(Yc.shape, dtype=bool)

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

        # optimization metrics
        loss_curve = [] # span loss
        grad_curve = [] # projected gradient norm
        slack_curve = [] # constraint slack
        angle_curve = [] # max angle
        gamma_curve = [] # line search step size

        # Extract indices for batched loss calculations
        ii, jj, kk = map(list, zip(*Ac))
        x_kk = X.T[kk]

        # optimization loop
        for update in range(num_updates):

            # calculate loss on joint-canonical adjacencies (batched)
            w_ii, w_jj = Wc[ii], Wc[jj]
            wiPk = w_ii - x_kk * (x_kk * w_ii).sum(axis=1, keepdims=True) / N
            wjPk = w_jj - x_kk * (x_kk * w_jj).sum(axis=1, keepdims=True) / N
            wiPkwi = (wiPk**2).sum(axis=1, keepdims=True)
            wjPkwj = (wjPk**2).sum(axis=1, keepdims=True)
            wiPkwj = (wiPk*wjPk).sum(axis=1, keepdims=True)

            wiPk_n, wjPk_n = wiPkwi**.5, wjPkwj**.5
            v_min_cos = (wiPkwj / (wiPk_n * wjPk_n)).min()

            v_loss = (wiPkwi * wjPkwj - wiPkwj**2).sum()

            # accumulate gradient
            grad_terms = 4 * (wiPk * wjPkwj - wiPkwj * wjPk)
            v_grad = np.zeros(Wc.shape)
            for idx, i in enumerate(ii): 
                v_grad[i] += grad_terms[idx]

            # # calculate loss and gradient on joint-canonical adjacencies (looped)
            # loss = 0.
            # min_cos = 1.
            # grad = np.zeros(Wc.shape)
            # for (i,j,k) in Ac:

            #     # get projected weights
            #     wi, wj, xk = Wc[i], Wc[j], X[:,k]
            #     Pk = PX[k]
            #     wiPk = wi @ Pk
            #     wjPk = wj @ Pk
            #     wiPkwi = wiPk @ wiPk
            #     wiPkwj = wiPk @ wjPk
            #     wjPkwj = wjPk @ wjPk

            #     # track minimum cosine
            #     wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)
            #     min_cos = min(min_cos, wiPkwj / (wiPk_n * wjPk_n))
    
            #     # accumulate span loss
            #     loss += wiPkwi * wjPkwj - wiPkwj**2
    
            #     # accumulate gradient
            #     grad[i] += 4 * (wiPk * wjPkwj - wiPkwj * wjPk)

            # assert np.isclose(v_loss, loss)
            # assert np.isclose(v_min_cos, min_cos)
            # assert np.allclose(v_grad, grad)

            loss, min_cos, grad = v_loss, v_min_cos, v_grad

            # # normalize by number of summands
            # loss /= len(Ac)
            # grad /= len(Ac)

            max_angle = np.arccos(min_cos)
            angle_curve.append(max_angle)
            loss_curve.append(loss)

            # # update active constraints
            # active = active | np.isclose((Wc @ X) * Yc, eps)

            # descent, not ascent
            Delta = -grad

            # project gradient onto active constraint planes
            for r in range(Delta.shape[0]):

                # # deactivate constraints satisfied by descent direction
                # # this is heuristic, not mathematically justified, cycles?
                # deactivate = ((Delta[r] @ (X[:, active[r]] * Yc[r, active[r]])) > 0)
                # if deactivate.any():
                #     # print(r, "deactivate!", deactivate.sum())
                #     active[r, active[r]] = ~deactivate

                # # keep W_lp plane always active
                # A_active = np.concatenate((X[:, active[r]], W_lp[r:r+1].T), axis=1)

                # skip projection when no constraints active
                if active[r].sum() == 0: continue

                A_active = X[:, active[r]]

                # orthonormal basis
                B_active = sl.orth(A_active)

                if B_active.shape[1] == N: input('active basis full rank...')

                # orthogonal projection onto null space of constraint normals
                Delta[r] = Delta[r] - B_active @ (B_active.T @ Delta[r])

            # norm of projected gradient (before clip scaling)
            pgnorm = norm(Delta)

            # clip step to inactive constraints: min_{clip >= 0} of (w + clip*d) @ x*y = eps
            clips = (eps - (Wc @ X) * Yc) / ((Delta @ X) * Yc)
            clip_scale = clips[~active & (0 <= clips) & (clips < np.inf)].min()
            # clip_scale = min(clip_scale, 1.) # avoids magnifying gradient
            Delta = clip_scale * Delta
            # print('clip', clip_scale)

            # analytical line search (batched)
            d_ii, d_jj = Delta[ii], Delta[jj]
            diPk = d_ii - x_kk * (x_kk * d_ii).sum(axis=1, keepdims=True) / N
            djPk = d_jj - x_kk * (x_kk * d_jj).sum(axis=1, keepdims=True) / N

            # accumulate span loss gradient at gamma = 1 (batched)
            vi, vj = wiPk + diPk, wjPk + djPk
            vjvj = (vj*vj).sum(axis=1, keepdims=True)
            vivj = (vi*vj).sum(axis=1, keepdims=True)
            step_grad_terms = 4 * (vi * vjvj - vivj * vj)
            v_step_grad = np.zeros(grad.shape) # sanity check gradient at end of step (gamma = 1)
            for idx, i in enumerate(ii): 
                v_step_grad[i] += step_grad_terms[idx]

            # accumulate cubic coefficients (batched)
            diPkdi = (diPk * diPk).sum(axis=1, keepdims=True)
            djPkdj = (djPk * djPk).sum(axis=1, keepdims=True)
            diPkdj = (diPk * djPk).sum(axis=1, keepdims=True)
            djPkdi = diPkdj

            wiPkdi = (wiPk * diPk).sum(axis=1, keepdims=True)
            wiPkdj = (wiPk * djPk).sum(axis=1, keepdims=True)
            wjPkdi = (wjPk * diPk).sum(axis=1, keepdims=True)
            wjPkdj = (wjPk * djPk).sum(axis=1, keepdims=True)
            diPkwj = wjPkdi
            wjPkwi = wiPkwj
            v_cubic = [
                np.sum(wjPkwj * wiPkdi - wjPkwi * wjPkdi),
                np.sum(2 * wjPkdj * wiPkdi + wjPkwj * diPkdi - wiPkwj * djPkdi - wjPkdi * (diPkwj + wiPkdj)),
                np.sum(2 * wjPkdj * diPkdi + wiPkdi * djPkdj - wjPkdi * diPkdj - djPkdi * (diPkwj + wiPkdj)),
                np.sum(djPkdj * diPkdi - djPkdi * djPkdi),
            ]

            # # analytical line search (looped)
            # cubic = [0, 0, 0, 0] # coefficients of line search derivative (cubic polynomial)
            # step_grad = np.zeros(grad.shape) # sanity check gradient at end of step (gamma = 1)
            # for (i,j,k) in Ac:

            #     # get weight and delta projections
            #     di, dj, wi, wj, xk = Delta[i], Delta[j], Wc[i], Wc[j], X[:,k]
            #     Pk = PX[k]
            #     wiPk = wi @ Pk
            #     wjPk = wj @ Pk
            #     diPk = di @ Pk
            #     djPk = dj @ Pk

            #     # accumulate span loss gradient at gamma = 1
            #     vi, vj = wiPk + diPk, wjPk + djPk
            #     step_grad[i] += 4 * (vi * (vj @ vj) - (vi @ vj) * vj)

            #     # accumulate cubic coefficients
            #     cubic[0] += wjPk @ (wjPk[:,np.newaxis] * wiPk - wiPk[:,np.newaxis] * wjPk) @ diPk
            #     cubic[1] += 2*(wjPk @ djPk)*(wiPk @ diPk) + (wjPk @ wjPk)*(diPk @ diPk) \
            #                 - (wiPk @ wjPk)*(djPk @ diPk) - (wjPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
            #     cubic[2] += 2*(wjPk @ djPk)*(diPk @ diPk) + (wiPk @ diPk)*(djPk @ djPk) \
            #                 - (wjPk @ diPk)*(diPk @ djPk) - (djPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
            #     cubic[3] += djPk @ (djPk[:,np.newaxis] * diPk - diPk[:,np.newaxis] * djPk) @ diPk

            # assert all([np.isclose(cub, v_cub) for (cub, v_cub) in zip(cubic, v_cubic)])

            cubic, step_grad = v_cubic, v_step_grad

            cubic = [4*cub for cub in cubic]

            # # normalize by number of summands
            # cubic = [cub/len(Ac) for cub in cubic]
            # step_grad /= len(Ac)

            # sanity check correct cubic coefficients at gamma = 0 and gamma = 1
            assert np.allclose(cubic[0], (grad * Delta).sum())
            assert np.allclose(sum(cubic), (step_grad * Delta).sum())

            # get extrema of line search objective in [0, 1)
            cubic = Polynomial(cubic)
            roots = cubic.roots()
            gammas = np.append(roots.real, (0.,1.,))
            gammas = gammas[(0 <= gammas) & (gammas <= 1)]
            # gammas = np.append(roots.real, (0.,))
            # gammas = gammas[(0 <= gammas) & (gammas < 1)]

            # evaluate extrema to find global min
            quartic = cubic.integ()
            evals = gammas[:,np.newaxis]**np.arange(5) @ quartic.coef
            gamma = gammas[evals.argmin()]

            # take step
            gamma_curve.append(gamma)
            Wc = Wc + gamma * Delta # stays in interior as long as 0 <= gamma < 1

            # calculate norm of projected gradient
            # pgnorm = np.fabs((grad * Delta).sum()) / norm(Delta)
            grad_curve.append(pgnorm)

            # update active constraints
            if gamma == 1.0: # went all the way to clip, some constraint has been activated
                active = active | np.isclose((Wc @ X) * Yc, eps)
            # print('num active', active.sum())

            # stop if infeasible, should not happen (but numerical issues when eps = 0)
            if eps > 0:
                feasible = (np.sign(Wc @ X) == Yc).all()
                if not feasible:
                    print("infeasible")
                    break
    
            # constraint slack
            slack = np.fabs(Wc @ X).min()
            slack_curve.append(slack)

            message = f"{update}/{num_updates}: loss={loss}, angle<={max_angle*180/np.pi}deg, |pgrad|={pgnorm**0.5}, slack={slack}, gamma={gamma}, nactive={active.sum()}"
            print(message)

            # stop if constraint padding violated
            padded = np.allclose(0, min(((Wc @ X) * Yc - eps).min(), 0))
            if not padded:
                print("padding violated", ((Wc @ X) * Yc - eps).min())
                break

            # check stopping criterion
            # early_stop = (gamma == 0.)
            early_stop = (pgnorm**.5 <= 1e-5) or (gamma <= 1e-15)
            # early_stop = (pgnorm**.5 <= 1e-5) or (gamma <= 1e-15) or (gamma == 1.)

            # save intermediate or final results
            if early_stop or (update + 1 == num_updates) or (update % save_period == 0):
                with open(f"span_opt_act_mp_{N}.pkl", "wb") as f:
                    pk.dump((Wc, loss_curve, slack_curve, grad_curve, angle_curve, gamma_curve), f)

            if early_stop:
                print("stopping early")
                break

    # load results
    with open(f"span_opt_act_mp_{N}.pkl", "rb") as f:
        (Wc, loss_curve, slack_curve, grad_curve, angle_curve, gamma_curve) = pk.load(f)

    # suppress large wall of text for N >= 6
    if N < 6:
        np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})

        print("\n" + "*"*8 + " change in weights " + "*"*8 + "\n")
    
        for i in range(len(Wc)):
            print(W_lp[i], Wc[i], np.fabs(Wc[i] @ X).min())
    
        print("\n" + "*"*8 + " span coefficients and residuals " + "*"*8 + "\n")
        print("wi ~ a * wj + b * xk, yik, resid, ijk")
        for (i, j, k) in Ac:
            ab = np.linalg.lstsq(np.vstack((Wc[j], X[:,k])).T, Wc[i], rcond=None)[0]
            resid = np.fabs(Wc[i] - (ab[0]*Wc[j] + ab[1]*X[:,k])).max()
            print(Wc[i], ab[0], Wc[j], ab[1], X[:,k], Yc[i,k], resid, i,j,k)

if __name__ == "__main__": main()


