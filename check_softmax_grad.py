"""
Take-aways:
the gradient direction through the softmax is the same as a desired direction directly in the simplex
but the magnitude varies for different logits that map to the same softmax output
gradient magnitudes are larger closer to the line x = y
intuitively:
    if one logit is much larger than the other, small changes have little effect
    when logits are all similar magnitudes, changes can have a bigger effect
simplex clipping in place of softmax might be more stable in terms of magnitude
"""

import torch as tr
import matplotlib.pyplot as pt

xy = tr.cartesian_prod(tr.linspace(-4, 4, 20), tr.linspace(-4, 4, 20)).requires_grad_()

g = {}
sm = tr.softmax(xy, dim=1)
# sm[:,0].sum().backward()
(sm[:,0] - sm[:,1]).sum().backward()
g[0] = xy.grad.clone().detach().numpy()
xy.grad *= 0

sm = tr.softmax(xy, dim=1)
# sm[:,1].sum().backward()
(sm[:,1] - sm[:,0]).sum().backward()
g[1] = xy.grad.clone().detach().numpy()

xy = xy.detach().numpy()
sm = sm.detach().numpy()

for i in range(2):

    pt.subplot(1,2,i+1)

    pt.plot(*xy.T, marker='.', linestyle='none', color='k')
    pt.plot(*sm.T, marker='.', linestyle='none', color='r')
    # # pt.quiver(*xy.T, *(sm - xy).T, angles='xy', scale_units='xy', scale=1, color='b')
    # for p in range(xy.shape[0]):
    #     pt.plot([xy[p,0], sm[p,0]], [xy[p,1], sm[p,1]], 'b-')
    pt.quiver(*xy.T, *g[i].T, angles='xy', scale_units='xy', scale=1, color='g')
   

pt.show()

