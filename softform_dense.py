import torch as tr
import softform as sf

# printing
def form_str(node):
    op, args = node

    if type(op) == str:
        return op

    if op.__name__ == "idleft":
        return form_str(args[0])
    if op.__name__ == "idright":
        return form_str(args[1])

    if op.__name__[-4:] == "left":
        return f"{op.__name__[:-4]}({form_str(args[0])})"
    if op.__name__[-5:] == "right":
        return f"{op.__name__[:-5]}({form_str(args[1])})"

    return f"{op.__name__}({','.join(map(form_str, args))})"

# unary ops
def minleft(x, y):
    return x.min(dim=-1, keepdims=True)[0].expand(*x.shape)

def minright(x, y):
    return y.min(dim=-1, keepdims=True)[0].expand(*x.shape)

def maxleft(x, y):
    return x.max(dim=-1, keepdims=True)[0].expand(*x.shape)

def maxright(x, y):
    return y.max(dim=-1, keepdims=True)[0].expand(*x.shape)

def meanleft(x, y):
    return x.mean(dim=-1, keepdims=True)[0].expand(*x.shape)

def meanright(x, y):
    return y.mean(dim=-1, keepdims=True)[0].expand(*x.shape)

def squareleft(x, y): return x**2
def squareright(x, y): return y**2

def negleft(x, y): return -x
def negright(x, y): return -y

# straight-through gradients for sign function
# based on https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/4
def softsign(x):
    return x + tr.sign(x).detach() - x.detach() # sign(x) in forward and x in backward

def signleft(x, y): return softsign(x)
def signright(x, y): return softsign(y)

def idleft(x, y): return x
def idright(x, y): return y

OPS = [
    idleft, idright, negleft, negright, signleft, signright, squareleft, squareright,
    minleft, minright, maxleft, maxright, meanleft, meanright,
    tr.add, tr.sub, tr.mul, tr.maximum, tr.minimum, sf.dotmean, sf.project, sf.reject,
]

class SoftFormDense(tr.nn.Module):

    # operations[k]: list of k-ary function handles
    # B: batch size
    def __init__(self, max_depth, logits=True, init_scale=None):
        super().__init__()

        if init_scale == None: init_scale = 1.0

        self.max_depth = max_depth
        self.logits = logits

        num_leaves = 2**max_depth
        num_inner = 2**max_depth - 1

        self.inners_attn = tr.nn.Parameter(init_scale*tr.randn(num_inner, len(OPS)))
        self.leaves_attn = tr.nn.Parameter(init_scale*tr.randn(num_leaves, 4)) # w, x, y, 1

    # batched inputs: 4 x B x N (4 inputs are w, x, y, 1)
    def forward(self, inputs):

        if self.logits:
            inners_attn = tr.softmax(self.inners_attn, dim=1)
            leaves_attn = tr.softmax(self.leaves_attn, dim=1)
        else:
            inners_attn = self.inners_attn
            leaves_attn = self.leaves_attn

        # process nodes by layer, deep to shallow

        # leaves
        values = (leaves_attn[:,:,None,None] * inputs).sum(dim=1) # leaves x B x N

        # inner nodes
        for depth in reversed(range(self.max_depth)):

            left, right = values[::2], values[1::2] # nodes x B x N
            results = tr.stack([op(left, right) for op in OPS]) # ops x nodes x B x N

            attn = inners_attn[2**depth-1:2**(depth+1)-1] # nodes x ops
            values = (attn.t()[:,:,None,None] * results).sum(dim=0) # nodes x B x N

        return values[0] # B x N

    """
    return formula tree for most attended ops
    """
    def harden(self):
        # leaf nodes
        nodes = {
            2**self.max_depth - 1 + n: ("wxy1"[tr.argmax(self.leaves_attn[n])], ())
            for n in range(2**self.max_depth)}
        # inner nodes
        for n in reversed(range(2**self.max_depth-1)):
            # left/right children
            nl, nr = 2*n + 1, 2*n + 2
            # most attended op
            op = OPS[tr.argmax(self.inners_attn[n])]
            # copy id ops directly for unbalanced trees
            if op == idleft: nodes[n] = nodes[nl]
            elif op == idright: nodes[n] = nodes[nr]
            else: nodes[n] = (op, (nodes[nl], nodes[nr]))
        return nodes[0]

if __name__ == "__main__":

    max_depth = 6

    # OPS = [
    #     idleft, idright, negleft, negright, signleft, signright, squareleft, squareright, 7
    #     minleft, minright, maxleft, maxright, meanleft, meanright, 13
    #     tr.add, tr.sub, tr.mul, tr.maximum, tr.minimum, sf.dotmean, sf.project, sf.reject, 21
    # ]
    model = SoftFormDense(max_depth, logits=False)

    # default to idleft and leaves 1
    inners_idx = [0]*(2**max_depth - 1)
    leaves_idx = [3]*(2**max_depth - 1)

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

    model.inners_attn.data[:] = 0
    model.inners_attn.data[tr.arange(len(inners_idx)), tr.tensor(inners_idx)] = 1.
    model.leaves_attn.data[:] = 0
    model.leaves_attn.data[tr.arange(len(leaves_idx)), tr.tensor(leaves_idx)] = 1.

    B, N = 2, 4
    inputs = tr.stack([ # 4 x B x n
        tr.ones(B,1) * tr.tensor([2.,1,1,0]), # w
        tr.ones(B,1) * tr.tensor([1.,1,-1,-1]), # x
        -tr.ones(B,N), # y
        tr.ones(B,N), # 1
    ])

    v = model(inputs)
    loss = v.sum()
    loss.backward()
    form = model.harden()

    print("perceptron?")
    print(form_str(form))
    print("output? [0, -1, 3, 2]")
    print(v)
    print("grads:")
    print(loss)
    print(model.inners_attn.grad)
    print(model.leaves_attn.grad)


