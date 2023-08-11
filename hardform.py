import torch as tr

if __name__ == "__main__":

    num_nodes = 2**6
    num_ops = 3
    hf = (0,) * num_nodes

    attn = tr.zeros(num_nodes, num_ops)
    attn[tr.arange(num_nodes), tr.tensor(hf)] = 1
    attn.requires_grad_(True)
    print(attn)

