import torch as tr

def to_tree(idx, ops):
    ops = [(arity, op) for arity in range(3) for op in ops[arity]]
    nodes = {}
    for n in reversed(range(len(idx))):
        # left/right children
        nl, nr = 2*n + 1, 2*n + 2
        # get current op
        arity, op = ops[idx[n]]
        # combine child nodes
        if arity == 0: nodes[n] = (op, ())
        if arity == 1: nodes[n] = (op, (nodes[nl],))
        if arity == 2: nodes[n] = (op, (nodes[nl], nodes[nr]))
    return nodes[0]

def to_attn(idx, ops):
    ops = [(arity, op) for arity in range(3) for op in ops[arity]]
    attn = tr.zeros(len(idx), len(ops))
    attn[tr.arange(len(idx)), tr.tensor(idx)] = 1.
    return attn

if __name__ == "__main__":

    from softform import form_str, form_eval, inv, square, dot

    max_depth = 6
    ops = {
        0: ['w', 'x', 'y', 'N', '1'],
        1: [tr.neg, tr.sign, inv, square],
        2: [tr.add, tr.mul, tr.maximum, tr.minimum, dot],
    }

    # perceptron rule:
    # w <- w + (y - sign(dot(w,x)))*x
    idx = [0] * (2**(max_depth + 1) - 1)
    idx[  0] =  9 # add(
    idx[  1] =  0 #     w,
    idx[  2] = 10 #     mul(
    idx[  5] =  9 #         add(
    idx[ 11] =  2 #             y,
    idx[ 12] =  5 #             neg(
    idx[ 25] =  6 #                 sign(
    idx[ 51] = 13 #                     dot(
    idx[103] =  0 #                         w,
    idx[104] =  1 #                         x)))),
    idx[  6] =  1 #         x))

    B, N = 2, 4
    inputs = {
        'w': tr.ones(B,1) * tr.tensor([2.,1,1,0]),
        'x': tr.ones(B,1) * tr.tensor([1.,1,-1,-1]),
        'y': -tr.ones(B,N),
        'N': N*tr.ones(B,N),
        '1': tr.ones(B,N),
    }

    tree = to_tree(idx, ops)
    attn = to_attn(idx, ops)
    v = form_eval(tree, inputs)

    print('attn', attn)
    print("perceptron?")
    print(form_str(tree))
    print("output? [0, -1, 3, 2]")
    print(v)


