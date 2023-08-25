import torch as tr

def to_tree(inners_idx, leaves_idx, ops):
    nodes = {}
    # leaves first
    for i, idx in enumerate(leaves_idx):
        n = i + len(leaves_idx) - 1 # node offset
        nodes[n] = ("wxy1"[idx], idx)
    for n in reversed(range(len(inners_idx))):
        # left/right children
        nl, nr = 2*n + 1, 2*n + 2
        # combine child nodes
        nodes[n] = (ops[inners_idx[n]], (nodes[nl], nodes[nr]))
    return nodes[0]

# batched inputs: 4 x B x N (4 inputs are w, x, y, 1)
def tree_eval(root, inputs):
    op, args = root
    # base case for leaves
    if type(op) == str: return inputs[args]
    # inner nodes
    arg0 = tree_eval(args[0], inputs)
    arg1 = tree_eval(args[1], inputs)
    return op(arg0, arg1)

def to_attn(inners_idx, leaves_idx, num_ops, num_inputs):
    if type(inners_idx) != tr.Tensor: inners_idx = tr.tensor(inners_idx)
    if type(leaves_idx) != tr.Tensor: leaves_idx = tr.tensor(leaves_idx)

    inners_attn = tr.zeros(len(inners_idx), num_ops)
    leaves_attn = tr.zeros(len(leaves_idx), num_inputs)

    inners_attn[tr.arange(len(inners_idx)), inners_idx] = 1.
    leaves_attn[tr.arange(len(leaves_idx)), leaves_idx] = 1.

    return inners_attn, leaves_attn

if __name__ == "__main__":

    import softform_dense as sfd
    import torch as tr

    max_depth = 6

    # default to idleft and leaves 1
    inners_idx = [0]*(2**max_depth - 1)
    leaves_idx = [3]*(2**max_depth)

    # perceptron rule:
    # w <- w + (y - sign(dot(w,x)))*x
    inners_idx[ 0] = 14 # add(
    leaves_idx[ 0] =  0 #     w,
    inners_idx[ 2] = 16 #     mul(
    inners_idx[ 5] = 15 #         sub(
    leaves_idx[32] =  2 #             y,
    inners_idx[12] =  4 #             sign(
    inners_idx[25] = 19 #                 dot(
    leaves_idx[40] =  0 #                     w,
    leaves_idx[42] =  1 #                     x)))),
    leaves_idx[48] =  1 #         x))

    B, N = 2, 4
    inputs = tr.stack([ # 4 x B x N
        tr.ones(B,1) * tr.tensor([2.,1,1,0]), # w
        tr.ones(B,1) * tr.tensor([1.,1,-1,-1]), # x
        -tr.ones(B,N), # y
        tr.ones(B,N), # 1
    ])

    tree = to_tree(inners_idx, leaves_idx, sfd.OPS)
    v = tree_eval(tree, inputs)

    print("perceptron?")
    print(sfd.form_str(tree))
    print("output? [0, -1, 3, 2]")
    print(v)

