import torch as tr
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt

np.set_printoptions(threshold=10e6)

if __name__ == "__main__":

    N = 5
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # get adjacencies
    A, K = adjacency(Y, sym=True)

    # wrap in tensors for grad opt
    W_lp = W
    W = (tr.tensor(W).float() + 0.01*tr.randn(W.shape)).requires_grad_()
    Y = tr.tensor(Y).float()
    X = tr.tensor(X).float()

    # form projection matrices
    # P[j] = Proj_X[:,j]
    Xt = X.t()
    P = tr.eye(N) - Xt.unsqueeze(1) * Xt.unsqueeze(2) / N

    # gradient update loop
    num_updates = 100
    lr = 0.00001
    loss_curve = []
    extr_curve = []
    for update in range(num_updates):

        # differentiate loss function
        losses = []
        for i in A:
            for j, k in zip(A[i], K[i]):
                pij = (tr.outer(W[i], W[j]) * P[k]).sum()
                pii = (tr.outer(W[i], W[i]) * P[k]).sum()
                pjj = (tr.outer(W[j], W[j]) * P[k]).sum()
                loss_ij = (pij**2 - pii*pjj)**2
                losses.append(loss_ij)
        loss = tr.stack(losses).sum()
        loss.backward()
        delta = -W.grad

        # project gradient to constraint set

        # update and zero gradient for next iter
        step_scale = lr
        # step_scale = lr / (np.log(update+1) + 1)
        W.data += delta * step_scale
        W.grad *= 0

        # stop if infeasible
        feasible = (tr.mm(W, X).sign() == Y).all().item()
        if not feasible:
            print("infeasible")
            break

        # check distance to feasible boundary
        extreme = tr.mm(W, X).abs().min()

        print(f"{update}/{num_updates}: loss={loss.item()}, extremality={extreme}, lr={step_scale}")
        loss_curve.append(loss.item())
        extr_curve.append(extreme.item())

    np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})

    W = W.detach().numpy()
    print(np.concatenate((W_lp, W), axis=1)[:10])

    for do_log in (False, True):
        pt.figure(figsize=(3,6))
        pt.subplot(2,1,1)
        pt.plot(loss_curve)
        pt.ylabel("loss")
        if do_log: pt.yscale('log')
        pt.subplot(2,1,2)
        pt.plot(extr_curve)
        pt.ylabel("constraint slack")
        if do_log: pt.yscale('log')
        pt.xlabel("update")
        pt.tight_layout()
        pt.show()    
