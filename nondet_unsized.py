# from collections.abc import Iterable, Sized # slow, native errors still informative
import random

sentinel = object()

def abyss(itr=(None,), choices=()):
    i = -1
    for i, item in enumerate(itr):
        if i > 0:
            for c in choices: yield c
        result = (yield item)
        if result != None:
            yield from abyss(iter(result), choices + (item,))
    if i < 0:
        yield sentinel

class NonDeterminator:

    def __init__(self):
        self.abyss = None

    def choice(self, itr):
        # if not isinstance(itr, Iterable): # slow, native errors still informative
        #     raise TypeError(f"Cannot choose from non-iterable {itr}")

        if self.abyss == None:
            # if not isinstance(itr, Sized): # slow, native errors still informative
            #     raise TypeError(f"Cannot choose from unsized iterable {itr}")
            # if len(itr) == 0: # slow, native errors still informative
            #     raise IndexError(f"Cannot choose from empty iterable {itr}")
            return random.choice(itr)

        else:
            item = self.abyss.send(itr)
            if item is sentinel:
                raise IndexError(f"Cannot choose from empty iterable {itr}")
            return item

    def runs(self, f):
        self.abyss = abyss()
        for _ in self.abyss: yield f()
        self.abyss = None
    
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
    nd.abyss = None

    print("\nnd.choice([])")
    try:
        nd.choice([])
    except:
        tb.print_exc()
    nd.abyss = None

    print("\nnd.choice((x for x in ()))")
    try:
        nd.choice((x for x in ()))
    except:
        tb.print_exc()
    nd.abyss = None

    print("\nfn(): return nd.choice(1)")
    def fn():
        return nd.choice(1)
    try:
        for ret in nd.runs(fn): print(ret)
    except:
        tb.print_exc()
    nd.abyss = None

    print("\nfn(): return nd.choice(range(0))")
    def fn():
        return nd.choice(range(0))
    try:
        for ret in nd.runs(fn): print(ret)
    except:
        tb.print_exc()
    nd.abyss = None

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

