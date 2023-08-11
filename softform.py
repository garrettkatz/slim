import torch as tr

# printing and evaluating formula trees
def form_str(node):
    op, args = node
    if type(op) == str: return op
    else: return f"{op.__name__}({','.join(map(form_str, args))})"

def form_eval(node, inputs):
    op, args = node
    if type(op) == str: return inputs[op]
    else: return op(*[form_eval(a, inputs) for a in args])   

# re-broadcasting dot-product
# assume x, y are shape (... x B x N)
def dot(x, y):
    return (x*y).sum(dim=-1, keepdims=True).expand(*x.shape)

# more stable for large N
def dotmean(x, y):
    return (x*y).mean(dim=-1, keepdims=True).expand(*x.shape)

def project(x, y):
    return dotmean(x, y) * y

def reject(x, y):
    return x * dotmean(y, y) - dotmean(x, y) * y

def min(x):
    return x.min(dim=-1, keepdims=True)[0].expand(*x.shape)

def max(x):
    return x.max(dim=-1, keepdims=True)[0].expand(*x.shape)

def mean(x):
    return x.mean(dim=-1, keepdims=True)[0].expand(*x.shape)

def inv(x):
    return tr.where(x == 0., 0., 1./x) # even if you define 1/x = 0 at x = 0, still not differentiable

def square(x):
    return tr.pow(x, 2)

def pow(x, y):
    return tr.where(tr.logical_and(x > 0., y > 0.), tr.pow(x, y), x)

class SoftForm(tr.nn.Module):

    # operations[k]: list of k-ary function handles
    # B: batch size
    def __init__(self, operations, max_depth, B, N, logits=True):
        super().__init__()

        self.max_depth = max_depth
        self.ops = operations
        self.logits = logits

        num_nodes = 2**(max_depth+1) - 1
        num_ops = sum(map(len, operations.values()))
        self.attention = tr.nn.Parameter(0.01*tr.randn(num_nodes, num_ops))

        self.z = tr.zeros(B,N) # placeholder for non-existent children

    def reset_dims(self, B, N):
        self.z = tr.zeros(B, N) # placeholder for non-existent children

    # batched inputs[v][b,:]: ND vector for variable v in example b
    def forward(self, inputs):

        if self.logits:
            attention = tr.softmax(self.attention, dim=1)
        else:
            attention = self.attention # average weights

        # process nodes by layer, deep to shallow

        # constants
        constants = []
        for op in self.ops[0]: constants.append(inputs[op])
        C = len(constants)
        constants = tr.stack(constants) # C x B x N

        # leaves
        attn = attention[2**self.max_depth-1:] # nodes x ops
        values = (attn[:,:C,None,None] * constants).sum(dim=1) # nodes x B x N

        for depth in reversed(range(self.max_depth)):

            left, right = values[::2], values[1::2] # nodes x B x N
            unary = [op(left) for op in self.ops[1]] # ops1 x nodes x B x N
            binary = [op(left, right) for op in self.ops[2]] # ops2 x nodes x B x N
            results = tr.stack(unary + binary) # ops1+2 x nodes x B x N

            attn = attention[2**depth-1:2**(depth+1)-1] # nodes x ops
            values = (attn[:,:C,None,None] * constants).sum(dim=1) # nodes x B x N

            values = values + (attn.t()[C:,:,None,None] * results).sum(dim=0) # nodes x B x N

        return values[0] # B x N


    """
    return formula tree for most attended ops
    """
    def harden(self):
        nodes = {}
        for n in reversed(range(self.attention.shape[0])):
            # left/right children
            nl, nr = 2*n + 1, 2*n + 2
            # most attended op
            if nl < self.attention.shape[0]: # has children
                most = tr.argmax(self.attention[n])
            else: # no children, limit to 0 arity ops
                most = tr.argmax(self.attention[n,:len(self.ops[0])])
            all_ops = [(arity, op) for arity in range(3) for op in self.ops[arity]]
            arity, op = all_ops[most]
            # combine child nodes
            if arity == 0: nodes[n] = (op, ())
            if arity == 1: nodes[n] = (op, (nodes[nl],))
            if arity == 2: nodes[n] = (op, (nodes[nl], nodes[nr]))
        return nodes[0]

if __name__ == "__main__":


    # perceptron rule:
    # w <- w + (y - sign(dot(w,x)))*x
    # add(
    #     w,
    #     mul(
    #         add(
    #             y,
    #             neg(
    #                 sign(
    #                     dot(
    #                         w,
    #                         x)))),
    #         x))
    # depth 6

    ops = {
        0: ['w', 'x', 'y', 'N', '1'],
        1: [tr.neg, tr.sign, inv, square],
        2: [tr.add, tr.mul, tr.maximum, tr.minimum, dot],
    }

    max_depth = 6
    B, N = 2, 4

    sf = SoftForm(ops, max_depth, B, N, logits=False)

    sf.attention.data[:] = 0
    sf.attention.data[  0, 9] = 1 # add(
    sf.attention.data[  1, 0] = 1 #     w,
    sf.attention.data[  2,10] = 1 #     mul(
    sf.attention.data[  5, 9] = 1 #         add(
    sf.attention.data[ 11, 2] = 1 #             y,
    sf.attention.data[ 12, 5] = 1 #             neg(
    sf.attention.data[ 25, 6] = 1 #                 sign(
    sf.attention.data[ 51,13] = 1 #                     dot(
    sf.attention.data[103, 0] = 1 #                         w,
    sf.attention.data[104, 1] = 1 #                         x)))),
    sf.attention.data[  6, 1] = 1 #         x))

    print(dot(tr.ones(B,N), tr.ones(B,N)))

    inputs = {
        'w': tr.ones(B,1) * tr.tensor([2.,1,1,0]),
        'x': tr.ones(B,1) * tr.tensor([1.,1,-1,-1]),
        'y': -tr.ones(B,N),
        'N': N*tr.ones(B,N),
        '1': tr.ones(B,N),
    }

    v = sf(inputs)
    loss = v.sum()
    loss.backward()
    form = sf.harden()

    print("perceptron?")
    print(form_str(form))
    print("output? [0, -1, 3, 2]")
    print(v)

    print(loss)
    print(sf.attention.data)
    print(sf.attention.grad)

    ops = {
        0: ['w', 'x'],
        1: [tr.neg, tr.sign, inv, square], # inv causes the NaNs
        2: [tr.add],
    }

    sf = SoftForm(ops, 1, B, N)

    sf.attention.data[:] = 0
    sf.attention.data[  0, 2] = 1 # add(
    sf.attention.data[  1, 0] = 1 #     w,
    sf.attention.data[  2, 1] = 1 #     x)

    v = sf(inputs)
    loss = v.sum()
    loss.backward()

    print("add(w, x):")
    print(loss)
    print(sf.attention.data)
    print(sf.attention.grad)


