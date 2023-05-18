This is the code for the paper

Anonymous et al. "Numerical Feasibility of One-Shot High-Capacity Learning in Linear Threshold Neurons."

You can regenerate the results with these steps:

1. Enumerate all the canonical hemichotomies with `python enumerate_ltms.py <N>`, where "<N>" is replaced with the maximum input dimension you want to enumerate.  If you leave out N, it will do all N from 3 to 8 inclusive.  On an 8-core machine, up to N=7 takes about 3 minutes, up to N=8 takes about half an hour.

1. Extract their adjacencies with `python canon_adjacent_ltms.py <N>`, again replacing "<N>" as desired.  On an 8-core machine, up to N=7 takes about 2 minutes.

1. Run `python sq_ccg_ltms_mp.py <N>` to do the conditional gradient span loss optimization for a particular value of N.  On an 8-core machine, N=4 takes a few seconds, and N=7 takes about 5 days.  This will save the optimization metrics and also print the data in the paper tables when it is finished.

Then you can generate the paper figures with these steps:

1. Run `python oneshot_visualization_pyvista.py` to generate the 3d visualization of continual one-shot learning

1. Run `python perceptron_baseline.py` to train perceptrons on all canonical hemichotomies up to N=8.

1. Run `python spanviol_ltms.py` to generate the span residual plot of un-optimized weights as returned by `enumerate_ltms.py`

1. Run `python plot_sq_ccg_ltms.py` to generate the optimization metrics plot

1. Run `python analyze_alpha_beta.py` to generate the alpha-beta scatter plot and beta-reflection plot


