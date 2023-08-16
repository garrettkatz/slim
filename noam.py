# based on https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer

class NoamScheduler:
    def __init__(self, base_lr, warmup, step_num=0):
        self.base_lr = base_lr
        self.warmup = warmup
        self.step_num = step_num

    def step(self):
        self.step_num += 1

    def get_lr(self):
        s = self.step_num + 1 # avoid 0 base in negative exponent
        lr = self.base_lr * min(s ** (-0.5), s * self.warmup ** (-1.5))
        return lr

    def apply_lr(self, optimizer):
        lr = self.get_lr()
        for p in optimizer.param_groups: p['lr'] = lr

if __name__ == "__main__":

    num_steps = 20000
    base_lr = .03
    warmup = 5000

    noam = NoamScheduler(base_lr, warmup)

    lrs = []
    for step in range(num_steps):
        lrs.append(noam.get_lr())
        noam.step()

    import matplotlib.pyplot as pt
    pt.plot(lrs)
    pt.show()

