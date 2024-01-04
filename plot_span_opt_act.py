import os
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mpl
import pickle as pk

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True

Ns = [4,5,6,7]
markers = '^sod' # markers for each N
num_pts = 10 # number of regularly spaced points plotted for each metric

fig, axs = pt.subplots(1, 4, figsize=(10, 2.75))

for m,N in zip(markers, Ns):

    # maybe larger runs aren't done yet, skip those
    fname = f"span_opt_act_mp_{N}.pkl"
    if not os.path.exists(fname): continue

    # load results
    with open(fname, "rb") as f:
        (Wc, log_period, loss_curve, slack_curve, grad_curve, angle_curve, gamma_curve) = pk.load(f)

    # remove oscillations around machine precision, looks bad on log scale
    meps = np.finfo(float).eps # machine precision
    loss_curve = np.array(loss_curve)
    # convergence = (loss_curve < meps).argmax() # first point (if any) under machine precision
    # if loss_curve[convergence] < meps: # if any,
    #     loss_curve = loss_curve[:convergence] # discard tail around machine precision

    # convert angles from rad to deg
    angle_curve = np.array(angle_curve) * 180/np.pi

    # plot metrics at regularly spaced points on a log scale
    # num_updates = len(loss_curve) * log_period
    # steps = np.geomspace(1, num_updates, num_pts).astype(int)
    # idxs = steps // log_period
    num_logged = len(loss_curve)
    idxs = np.geomspace(1, num_logged, num_pts).astype(int) - 1
    steps = idxs * log_period + 1

    # print(idxs[-1], grad_curve[idxs[-1]], num_logged, grad_curve[-1])

    # print final metrics
    print(f"N={N}: loss->{loss_curve[-1]}, |pgrad|->{grad_curve[-1]}, slack->{slack_curve[-1]}, ang->{angle_curve[-1]}")

    # Loss
    pt.sca(axs[0])
    pt.plot(steps, loss_curve[idxs], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Span Loss")
    pt.yscale('log')
    pt.xscale('log')
    pt.ylabel("Metric value")
    axs[0].tick_params(axis='y', which='minor', right=True, labelleft=False)
    axs[0].tick_params(axis='y', which='major', right=True, labelright=False)
    axs[0].yaxis.get_minor_locator().set_params(numticks=25, subs=[1])

    # Projected gradient norm
    pt.sca(axs[1])
    pt.plot(steps, np.array(grad_curve)[idxs], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Projected Gradient Norm")
    pt.yscale('log')
    pt.xscale('log')
    axs[1].tick_params(axis='y', which='minor', right=True, labelleft=False)
    axs[1].tick_params(axis='y', which='major', right=True, labelright=False)
    axs[1].yaxis.get_minor_locator().set_params(numticks=25, subs=[1])

    # Constraint slack
    pt.sca(axs[2])
    pt.plot(steps, np.array(slack_curve)[idxs], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Constraint Slack")
    pt.xscale('log')
    axs[2].tick_params(axis='y', which='major', right=True)

    # Maximum angle
    pt.sca(axs[3])
    pt.plot(steps, angle_curve[idxs], f'k{m}--', mfc='w', label=f"$N={N}$")
    pt.title("Maximum Angle (deg)")
    pt.yscale('log')
    pt.xscale('log')
    axs[3].tick_params(axis='y', which='minor', right=True)
    axs[3].tick_params(axis='y', which='major', right=True, labelright=True)
    axs[3].yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])

# final formatting    
pt.sca(axs[2])
pt.legend(loc='upper right')
fig.supxlabel("Optimization Step")
pt.tight_layout()

# save and show
pt.savefig(f"sq_ccg_ltm_results.pdf")
pt.show()

