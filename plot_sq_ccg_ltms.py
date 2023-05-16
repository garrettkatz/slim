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

fig, axs = pt.subplots(1, 4, figsize=(10, 2.75))

for m,N in zip(markers, Ns):
    # print(N)

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
    # print(steps[-1], sq_loss_curve[-1], cos_curve[-1], np.arccos(1 - cos_curve[-1]), np.arccos(1 - cos_curve[-1])*180/np.pi)
    # print(steps[-1], sq_loss_curve[steps[-1]-1], cos_curve[steps[-1]-1], np.arccos(1 - cos_curve[steps[-1]-1]), np.arccos(1 - cos_curve[steps[-1]-1])*180/np.pi)

    print(f"N={N}: sqloss->{sq_loss_curve[-1]}, pgn->{pgn_curve[-1]}, slack->{extr_curve[-1]}, ang->{np.arccos(1 - cos_curve[-1])*180/np.pi}")
        
    pt.sca(axs[0])
    pt.plot(steps, sq_loss_curve[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Span Loss")
    pt.yscale('log')
    pt.xscale('log')
    pt.ylabel("Metric value")
    # axs[0].tick_params(axis='y', which='minor', right=True)
    # axs[0].tick_params(axis='y', which='major', right=True)
    # axs[0].yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    axs[0].tick_params(axis='y', which='minor', right=True, labelleft=False)
    axs[0].tick_params(axis='y', which='major', right=True, labelright=False)
    axs[0].yaxis.get_minor_locator().set_params(numticks=25, subs=[1])
    
    pt.sca(axs[1])
    pt.plot(steps, np.array(pgn_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Projected Gradient Norm")
    pt.yscale('log')
    pt.xscale('log')
    # axs[1].tick_params(axis='y', which='minor', right=True)
    # axs[1].tick_params(axis='y', which='major', right=True)
    # axs[1].yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    axs[1].tick_params(axis='y', which='minor', right=True, labelleft=False)
    axs[1].tick_params(axis='y', which='major', right=True, labelright=False)
    axs[1].yaxis.get_minor_locator().set_params(numticks=25, subs=[1])

    pt.sca(axs[2])
    pt.plot(steps, np.array(extr_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Constraint Slack")
    # pt.yscale('log')
    pt.xscale('log')
    # axs[2].tick_params(axis='y', which='minor', right=True)
    axs[2].tick_params(axis='y', which='major', right=True)
    # axs[2].yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    
    pt.sca(axs[3])
    # pt.plot(steps, np.array(cos_curve)[steps-1], f'k{m}--', mfc='w', label=f"$N={N}$")
    # pt.title("max 1 - cos")
    pt.plot(steps, np.arccos(1 - np.array(cos_curve))[steps-1] * 180/np.pi, f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Maximum Angle (deg)")
    pt.yscale('log')
    pt.xscale('log')
    axs[3].tick_params(axis='y', which='minor', right=True)
    axs[3].tick_params(axis='y', which='major', right=True, labelright=True)
    axs[3].yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    
pt.sca(axs[2])
pt.legend(loc='upper right')

fig.supxlabel("Optimization Step")
# fig.suptitle("Metric")
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

