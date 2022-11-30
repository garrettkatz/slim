# from collections.abc import Iterable, Sized # slow, native errors still informative
import random

class NonDeterminator:

    def __init__(self):
        self.counters = None
        self.choices = None
        self.depth = None

    def choice(self, itr):

        if self.depth == None:
            return random.choice(itr)

        if self.depth == len(self.counters):
            self.counters.append(0)
            self.choices.append(tuple(itr))

        i = self.counters[self.depth]
        item = self.choices[self.depth][i]
        self.depth += 1
        
        return item

    def runs(self, f):
        self.counters = []
        self.choices = []

        while True:
            self.depth = 0
            yield f()

            for c in range(self.depth-1, -1, -1):
                self.counters[c] += 1
                if self.counters[c] < len(self.choices[c]): break
                self.counters.pop()
                self.choices.pop()
            if len(self.counters) == 0: break

        self.depth = None

    def counter_string(self):
        return " ".join(f"{c}:{co}/{len(ch)}"
            for c,(co,ch) in enumerate(zip(self.counters, self.choices)) if len(ch) > 1)
    
if __name__ == "__main__":

    import itertools as it
    import traceback as tb

    nd = NonDeterminator()

    # one choice
    def fn():
        return nd.choice(range(3))

    assert 0 <= fn() < 3
    assert tuple(nd.runs(fn)) == tuple(range(3))

    # multi choice
    def fn():
        x = nd.choice(range(3))
        y = nd.choice(range(4))
        z = nd.choice(range(5))
        return x, y, z

    assert (0, 0, 0) <= fn() < (3, 4, 5)
    assert tuple(nd.runs(fn)) == tuple(it.product(range(3), range(4), range(5)))

    # multi choice with branching
    def fn():
        x = nd.choice(range(3))
        y = nd.choice(range(4))
        if y < 2:
            z = nd.choice(range(5))
        else:
            z = nd.choice(range(2))
        return x, y, z

    target = tuple((x,) + yz for x,yz in it.product(range(3),
        tuple(it.product(range(2), range(5))) + tuple(it.product(range(2,4), range(2)))))
    assert tuple(nd.runs(fn)) == target

    # choice dependencies
    def fn():
        x = nd.choice(range(3))
        y = nd.choice(range(x+1, 4))
        z = nd.choice(range(y+1, 5))
        return x, y, z

    target = tuple(it.combinations(range(5),3))
    assert tuple(nd.runs(fn)) == target

    # errors
    print("\nnd.choice(1)")
    try:
        nd.choice(1)
    except:
        tb.print_exc()

    print("\nnd.choice([])")
    try:
        nd.choice([])
    except:
        tb.print_exc()

    print("\nnd.choice((x for x in ()))")
    try:
        nd.choice((x for x in ()))
    except:
        tb.print_exc()

    print("\nfn(): return nd.choice(1)")
    def fn():
        return nd.choice(1)
    try:
        for ret in nd.runs(fn): print(ret)
    except:
        tb.print_exc()

    print("\nfn(): return nd.choice(range(0))")
    def fn():
        return nd.choice(range(0))
    try:
        for ret in nd.runs(fn): print(ret)
    except:
        tb.print_exc()
    
    print("Thrown errors shown above")

    # slim-inspired test
    M = 2 # >= 3 is oom
    N = 2 # >= 3 is oom
    def fn():
        h = ()
        for j,k in it.permutations(range(M), 2):
            hjk = ()
            for m in range(M):
                hjkm = nd.choice(range(2**N))
                hjk += (hjkm,)
            h += (hjk,)
        return h

    target = tuple(it.product(tuple(it.product(range(2**N), repeat=M)), repeat=M*(M-1)))
    assert tuple(nd.runs(fn)) == target
    
    # print("\nslim:")
    # for h in nd.runs(fn):
    #     print("---")
    #     for hjk in h: print("", hjk)

    from time import perf_counter

    def fn():
        x = ()
        for i in range(5):
            x += (nd.choice(range(10)),)
        return x

    start = perf_counter()
    for ret in nd.runs(fn):
        pass
    print(f"{perf_counter() - start} seconds")

    # count tree test
    def fn():
        nodes = 1
        for d in range(3):
            numc = [nd.choice(range(3)) for _ in range(nodes)]
            nodes = sum(numc)
        return nodes
    results = tuple(nd.runs(fn))
    print(len(results))
    assert len(results) == 1 + 13 + 13**2
