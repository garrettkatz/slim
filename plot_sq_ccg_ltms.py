import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mpl
import pickle as pk

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True

Ns = [4,5,6,7]
markers = '^sod'
num_pts = 10
postfix = "_ana"

fig, axs = pt.subplots(1, 4, figsize=(10, 2.5))

for m,N in zip(markers, Ns):
    print(N)

    with open(f"sq_ccg_ltm_mp_{N}{postfix}.pkl", "rb") as f:
        (Wc, sq_loss_curve, loss_curve, extr_curve, gn_curve, pgn_curve, cos_curve, scale_curve) = pk.load(f)

    # remove oscillations around machine precision
    sq_loss_curve = np.array(sq_loss_curve)
    meps = np.finfo(float).eps
    convergence = (sq_loss_curve < meps).argmax()
    if sq_loss_curve[convergence] < meps:
        sq_loss_curve = sq_loss_curve[:convergence]

    num_updates = len(sq_loss_curve)
    # step = num_updates // num_pts
    # steps = np.arange(0, num_updates, step)+1
    steps = np.geomspace(1, num_updates, num_pts).astype(int)
        
    pt.sca(axs[0])
    pt.plot(steps, sq_loss_curve[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Span Loss")
    pt.yscale('log')
    pt.xscale('log')
    pt.ylabel("Metric")
    
    pt.sca(axs[1])
    pt.plot(steps, np.array(pgn_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Projected Gradient Norm")
    pt.yscale('log')
    pt.xscale('log')

    pt.sca(axs[2])
    pt.plot(steps, np.array(extr_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Constraint Slack")
    # pt.yscale('log')
    pt.xscale('log')
    
    pt.sca(axs[3])
    # pt.plot(steps, np.array(cos_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    # pt.title("max 1 - cos")
    pt.plot(steps, np.arccos(1 - np.array(cos_curve))[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Maximum Angle (rad)")
    pt.yscale('log')
    pt.xscale('log')
    
pt.sca(axs[2])
pt.legend(loc='upper right')

fig.supxlabel("Optimization Step")
pt.tight_layout()
pt.savefig(f"sq_ccg_ltm_results.pdf")
pt.show()


# fig, axs = pt.subplots(3, len(Ns), figsize=(15, 5))

# for n,N in enumerate(Ns):

#     with open(f"ccg_ltm_{N}.pkl", "rb") as f:
#         (Wc, loss_curve, extr_curve, gn_curve, pgn_curve) = pk.load(f)

#     num_updates = len(loss_curve)
    
#     pt.sca(axs[0,n])
#     pt.plot(loss_curve, 'k-')
#     if n == 0: pt.ylabel("Span\nLoss")
#     pt.yscale('log')
#     pt.title(f"$N={N}$")
#     pt.xticks([], [])

#     pt.sca(axs[1,n])
#     # pt.plot(gn_curve, 'k:')
#     pt.plot(pgn_curve, 'k-')
#     if n == 0: pt.ylabel("Gradient\nNorm")
#     pt.yscale('log')
#     pt.xticks([], [])

#     pt.sca(axs[2,n])
#     pt.plot(extr_curve, 'k-')
#     if n == 0: pt.ylabel("Constraint\nSlack")
#     pt.yscale('log')

# fig.supxlabel("Optimization Step")
# pt.tight_layout()
# pt.savefig(f"ccg_ltm_results.pdf")
# pt.show()

