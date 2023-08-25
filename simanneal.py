import numpy as np
import torch as tr
import hardform_dense as hfd
import softform_dense as sfd
from softfit_dense import VecRule, load_examples

if __name__ == "__main__":

    num_itrs = 1
    use_simplex_clipping = True

    num_ops = len(sfd.OPS)
    num_inp = 4
    max_depth = 3

    num_inner = 2**max_depth - 1
    num_leaves = 2**max_depth
    num_nodes = num_inner + num_leaves

    # initialize examples
    examples, optimal_loss = load_examples(list(range(3,5)))

    # initialize softform
    model = VecRule(max_depth, logits=False)
    inners_attn = model.sf.inners_attn
    leaves_attn = model.sf.leaves_attn

    def get_loss():
        total_loss = tr.tensor(0.)

        for (inputs, (w_new, mdenom)) in examples:
            w_pred = model(inputs)
            cosgam = (w_new*w_pred).sum(dim=1) # already normalized
            sinmarg = tr.cos(mdenom)
            total_loss += -(cosgam / sinmarg).mean() / len(examples)

        return total_loss

    # initialize candidate
    inners_idx = tr.randint(num_ops, size=(num_inner,))
    leaves_idx = tr.randint(num_inp, size=(num_leaves,))
    # print(inners_idx, leaves_idx)

    cands, losses, temps, probs = [], [], [], []
    new_candidate = True
    for itr in range(num_itrs):

        # get new candidate gradient over transitions
        if new_candidate:
            new_candidate = False

            # soften current candidate
            ia, la = hfd.to_attn(inners_idx, leaves_idx, num_ops, num_inp)
            inners_attn.data, leaves_attn.data = ia, la
            model.zero_grad()

            # get gradients
            cand_loss = get_loss()
            # print('cl', cand_loss)
            cand_loss.backward()

            clipped = []
            gradscales = []
            print(inners_idx, leaves_idx)
            for attn in (model.sf.inners_attn, model.sf.leaves_attn):
                # project away grad components orthogonal to attention simplices
                print('g', attn.grad.min(), attn.grad.max())
                print('dat',attn.data[:1])
                print('gra',attn.grad[:1])
                attn.grad -= attn.grad.mean(dim=1, keepdim=True)
                print('gpost', attn.grad.min(), attn.grad.max())
                print('gra',attn.grad[:1])
                # clip negative grads to limits of attention simplices
                # # these were clipping in positive grad direction!
                # minpos = tr.min(tr.where(attn.grad > 0, (1 - attn.data) / attn.grad, 1), dim=1, keepdim=True).values
                # minneg = tr.min(tr.where(attn.grad < 0, (  - attn.data) / attn.grad, 1), dim=1, keepdim=True).values
                minpos = tr.min(tr.where(attn.grad > 0, (attn.data    ) / attn.grad, 1), dim=1, keepdim=True).values
                minneg = tr.min(tr.where(attn.grad < 0, (attn.data - 1) / attn.grad, 1), dim=1, keepdim=True).values
                grad_scale = tr.minimum(tr.minimum(minpos, minneg), tr.ones(minpos.shape))
                attn.grad *= grad_scale
                # print(attn.grad)
                print(grad_scale)
                print('gppost', attn.grad.min(), attn.grad.max())
                clipped += (grad_scale != 1.).tolist()
                gradscales += grad_scale.tolist()
            clipped = np.mean(clipped)
            gradscale = np.mean(gradscales)
            print('gs', gradscale)

            # extract probability distributions at simplex boundary
            # print((inners_attn.data - inners_attn.grad).min())
            # print((leaves_attn.data - leaves_attn.grad).min())
            # print((inners_attn.data - inners_attn.grad).max())
            # print((leaves_attn.data - leaves_attn.grad).max())
            print((inners_attn.grad).min())
            print((leaves_attn.grad).min())
            print((inners_attn.grad).max())
            print((leaves_attn.grad).max())
            inners_attn_prob = tr.clamp(inners_attn.data - inners_attn.grad, 0., 1.)
            leaves_attn_prob = tr.clamp(leaves_attn.data - leaves_attn.grad, 0., 1.)

            # print(tr.isnan(model.sf.inners_attn.data).any())
            # print(tr.isnan(model.sf.inners_attn.grad).any())
            # print(tr.isnan(model.sf.leaves_attn.data).any())
            # print(tr.isnan(model.sf.leaves_attn.data).any())
            # print("ia", ia)
            # print("id",model.sf.inners_attn.data)
            # print("ig",model.sf.inners_attn.grad)
            # print("ip",inners_attn_prob)
            # print("la",la)
            # print("ld",model.sf.leaves_attn.data)
            # print("lg",model.sf.leaves_attn.grad)
            # print("lp",leaves_attn_prob)

        # sample next candidate
        inners_next = tr.multinomial(inners_attn_prob, num_samples=1).flatten()
        leaves_next = tr.multinomial(leaves_attn_prob, num_samples=1).flatten()
        # print('next',inners_next, leaves_next)
        inners_attn.data, leaves_attn.data = hfd.to_attn(inners_next, leaves_next, num_ops, num_inp)
        with tr.no_grad(): next_loss = get_loss()
        losses.append(next_loss.item())
        cands.append(cand_loss.item())

        # simulated annealing update
        loss_diff = (next_loss - cand_loss).item()
        # temperature = (num_itrs - itr) / num_itrs
        temperature = 1. / (np.log(1 + itr) + 1)
        temps.append(temperature)
        pr = 1. if loss_diff < 0 else np.exp(-loss_diff / temperature)
        # pr = 1. if loss_diff < 0 else 0.
        probs.append(pr)
        if np.random.rand() < pr:
            inners_idx, leaves_idx = inners_next, leaves_next
            new_candidate = True

        print(f"{itr} of {num_itrs}: loss={next_loss.item():+.3f} vs {cand_loss.item():+.3f} {'*'*8 if new_candidate else ''}")
    
    import matplotlib.pyplot as pt
    pt.subplot(3,1,1)
    pt.plot(losses, 'b-')
    pt.plot(cands, 'g-')
    pt.ylabel("Loss")
    pt.subplot(3,1,2)
    pt.plot(temps)
    pt.ylabel("Temp")
    pt.subplot(3,1,3)
    pt.plot(probs)
    pt.ylabel("Prob")
    pt.xlabel("Iter")
    pt.show()
            
