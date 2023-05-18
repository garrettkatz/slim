This is the code for the paper

Anonymous et al. "Numerical Feasibility of One-Shot High-Capacity Learning in Linear Threshold Neurons."

You can regenerate the results with these steps:

1. Enumerate all the canonical hemichotomies with `python enumerate_ltms.py <N>`, where "<N>" is replaced with the maximum input dimension you want to enumerate.  If you leave out N, it will do all N from 3 to 8 inclusive (takes about half an hour on an 8-core machine).

1. Extract their adjacencies with `python canon_adjacent_ltms.py <N>`, again replacing "<N>" as desired

1. Run `python sq_ccg_ltms_mp.py` to do the conditional gradient span loss optimization for N=4, it should take a few seconds on an 8-core machine.  For larger N, uncomment the appropriate block around lines 56-78.  This will also print the data in the paper tables when it is finished.

Then you can generate the paper figures with these steps:

1. Run `python oneshot_visualization_pyvista.py` to generate the 3d visualization of continual one-shot learning

1. Run `python perceptron_baseline.py` to generate the perceptron sample efficiency plot, may take several minutes

1. Run `python spanviol_ltms.py` to generate the span residual plot of un-optimized weights as returned by `enumerate_ltms.py`

1. Run `python plot_sq_ccg_ltms.py` to generate the optimization metrics plot

1. Run `python analyze_alpha_beta.py` to generate the alpha-beta scatter plot and beta-reflection plot


