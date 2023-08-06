import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mpl
import pickle as pk

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True

Ns = [4,5,6]
markers = '^so'
num_pts = 10

fig, axs = pt.subplots(1, 3, figsize=(8, 2.5))

for m,N in zip(markers, Ns):

    with open(f"ccg_ltm_{N}.pkl", "rb") as f:
        (Wc, loss_curve, extr_curve, gn_curve, pgn_curve) = pk.load(f)

    num_updates = len(loss_curve)
    step = num_updates // num_pts
    steps = np.arange(0, num_updates, step)
        
    pt.sca(axs[0])
    pt.plot(steps, loss_curve[::step], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.ylabel("Span Loss")
    pt.yscale('log')

    pt.sca(axs[1])
    # pt.plot(steps, gn_curve[::step], 'k:')
    # pt.plot(steps, np.array(pgn_curve[::step])**0.5, f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.plot(steps, pgn_curve[::step], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.ylabel("Projected Gradient Norm")
    pt.yscale('log')

    pt.sca(axs[2])
    pt.plot(steps, extr_curve[::step], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.ylabel("Constraint Slack")
    pt.yscale('log')

pt.sca(axs[0])
pt.legend()

fig.supxlabel("Optimization Step")
pt.tight_layout()
pt.savefig(f"ccg_ltm_results.pdf")
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

