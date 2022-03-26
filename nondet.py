"""
This might all be doable more elegantly using coroutines/yield from:
https://medium.com/analytics-vidhya/python-generators-coroutines-async-io-with-examples-28771b586578
https://stackoverflow.com/questions/9708902/in-practice-what-are-the-main-uses-for-the-yield-from-syntax-in-python-3-3
https://peps.python.org/pep-0342/
https://peps.python.org/pep-0380/#formal-semantics
"""
class Runner:
    def __init__(self):
        self.iterators = []
        self.items = []
        self.depth = 0
    def run(self, fn):
        while True:
            self.depth = 0
            try:
                yield fn()
            except StopIteration:
                pass
            carry = self.depth - 1
            while carry >= 0:
                try:
                    self.items[carry] = next(self.iterators[carry])
                    break
                except StopIteration:
                    self.iterators = self.iterators[:carry]
                    self.items = self.items[:carry]
                    carry -= 1
            if carry == -1: return
            
    def choice(self, iterable):
        # how to check if iteration already started over iterable at called line?
        # should be based on caller's frame, if it was already constructed in that frame and line
        # if you increment an outer-more counter, all inner-mores should be already finished and erased?

        if self.depth < len(self.items):
            item = self.items[self.depth]
        else:
            iterator = iter(iterable)
            item = next(iterator)
            self.iterators.append(iterator)
            self.items.append(item)
        self.depth += 1
        return item
        
if __name__ == "__main__":

    # # works, just ugly impl of Runner
    # r = Runner()
    # def fn():
    #     x = r.choice(range(3))
    #     if x != 1:
    #         y = r.choice(range(3))
    #     else:
    #         # y = r.choice(range(4))
    #         y = r.choice([])
    #     z = r.choice([True, False])
    #     return x,y,z

    # for ret in r.run(fn):
    #     print(ret)
    #     # if x == 3: break


    # # coro scratch 1
    # def coro1():
    #     x = (yield 1)
    #     print('coro1', x)
    #     yield 3
    #     yield 4

    # def coro2(c):
    #     z = c.send(None)
    #     print('coro2 z', z)
    #     # y = c.send(2)
    #     # print('coro2 y', y)
    #     for y in c:
    #         print('coro2 y', y)
    
    # c = coro1()
    # print(c)
    # coro2(c)

    # # coro scratch 2
    # def coro():
    #     for i in range(4):
    #         yield i

    # c = coro()
    # print(c.send(None))
    # print(c.send(None))
    # print(c.send(None))
    # print(c.send(None))

    # # coro scratch 3: generator with side yields    
    # def coro():
    #     x = (yield)
    #     for i in range(4):
    #         x = (yield x + 2)
    # c = coro()
    # c.send(None)    
    # def gen():
    #     for i in range(4):
    #         yield c.send(i)    
    # for i in gen():
    #     print(i)

    # # coro scratch 4: dynamically bound coro, multiple callers
    # def coro():
    #     x = (yield)
    #     for i in range(8):
    #         x = (yield x + 2)
    # c = None
    # def gen1():
    #     for i in range(4):
    #         yield c.send(i)
    # def gen2():
    #     for i in range(4):
    #         yield c.send(i)
    # c = coro()
    # c.send(None)    
    # for i in gen1(): print(i)
    # for i in gen2(): print(i)
    
    # # coro scratch 5:
    # # g = (yield from range(4)) # "'yield' outside function" syntax error
    # def gen(): yield from range(4)    
    # g = gen()    
    # for i in g: print(i)
    
    # # nd coro scratch 1
    # depth = -1
    # itrs = []
    # itms = []
    # def choi(itr):
    #     global depth, itrs, itms
    #     depth += 1
    #     if depth >= len(itrs):
    #         itrs.append(iter(itr))
    #         itms.append(next(itrs[depth]))
    #     return itms[depth]    
    # def fn():
    #     global depth, itrs, itms
    #     print("fn:::")
    #     print(depth)
    #     print(itrs)
    #     print(itms)
    #     input("..")
    #     x = choi(range(3))
    #     y = choi(range(2))
    #     return x,y
    # def abyss(d):
    #     global depth, itrs, itms
    #     print((" "*d) + f"abyss({d}):::")
    #     print((" "*d) + f"{depth}")
    #     print((" "*d) + f"{itrs}")
    #     print((" "*d) + f"{itms}")
    #     if d < len(itrs):
    #         yield from abyss(d+1)
    #         for itm in itrs[d]:
    #             itms[d] = itm
    #             yield from abyss(d+1)
    #         itrs, itms = itrs[:d], itms[:d]
    #     else:
    #         depth = -1
    #         yield fn()            
    # def run():
    #     global depth, itrs, itms
    #     depth = -1
    #     yield fn()
    #     yield from abyss(0)
    # results = []
    # for result in run():
    #     print("result:::")
    #     print(result)
    #     results.append(result)
    # print(results)

    # depth = -1
    # print(fn())
    # print(itms)
    # depth = -1
    # print(fn())
    # print(itms)
    # depth = -1
    # print(fn())
    # print(itms)

    # # nd coro scratch 2
    # def choi(itr, a):
    #     i = a.send(itr)
    #     return i
    
    # def abyss():
    #     itr = (yield) # from choice
    #     for i in itr:
    #         yield i # to choice

    # a = abyss()
    # a.send(None)

    # def fn():
    #     x = choi(range(3), a)
    #     return x
    
    # # works with try-except
    # try:
    #     while True:
    #         print(fn())
    # except StopIteration:
    #     print("Done.")

    # # # still uncaught stopiter
    # # def runs(f):
    # #     while True: yield fn()
    # # for res in runs(fn):
    # #     print(res)
    # # print("Done.")

    # # # 3 good then stop iter
    # # result = fn()
    # # print(result)

    # # result = fn()
    # # print(result)

    # # result = fn()
    # # print(result)

    # # # stopiter
    # # result = fn()
    # # print(result)

    # # nd coro scratch 3: no recursion, works with no try-except (unless itr is empty)
    # # x = yield z: x < ... callers ... < z
    # def abyss(): # < run
    #     itr = iter( (yield) ) # itr < choi < run < _
    #     yield next(itr) # _ < run < choi < i
    #     for i in itr:
    #         yield # _ < choi < run < _
    #         yield i # _ < run < choi < i

    # def run(f, a):
    #     for _ in a:
    #         print(f())

    # def choi(itr, a):
    #     i = a.send(itr)
    #     return i

    # a = abyss()
    # def fn():
    #     x = choi(range(4), a)
    #     return x
    # run(fn, a)

    # # nd coro scratch 4: no recursion but multichoice
    # def abyss(): # < run
    #     carry = -1
    #     itr0 = iter( (yield) )
    #     for i in itr0:
    #         if carry > -1: yield
    #         itr1 = iter( (yield i) )
    #         for j in itr1:
    #             if carry > 0: yield
    #             if carry > 0: yield i
    #             itr2 = iter( (yield j) )
    #             for k in itr2:
    #                 if carry > 1: yield
    #                 if carry > 1: yield i
    #                 if carry > 1: yield j
    #                 base = yield k
    #                 carry = 2
    #             carry = 1
    #         carry = 0

    # def run(f, a):
    #     for _ in a:
    #         print("f() =", f())

    # def choi(itr, a):
    #     i = a.send(itr)
    #     return i

    # a = abyss()
    # def fn():
    #     x = choi(range(2), a)
    #     y = choi(range(2), a)
    #     z = choi(range(2), a)
    #     return x, y, z
    # run(fn, a)

    # # nd coro scratch 5: recursive multichoice
    # # def abyss(itr=(None,), choices=()):
    # #     first_done = False
    # #     for i in itr:
    # #         if first_done:
    # #             for c in choices: yield c
    # #         result = (yield i)
    # #         if result != None:
    # #             yield from abyss(iter(result), choices + (i,))
    # #         first_done = True
    # def abyss(itr=(None,), choices=()):
    #     for i, item in enumerate(itr):
    #         if i > 0:
    #             for c in choices: yield c
    #         result = (yield item)
    #         if result != None:
    #             yield from abyss(iter(result), choices + (item,))

    # def run(f, a):
    #     for _ in a:
    #         res = f()
    #         # print("f() =", res)

    # def choi(itr, a):
    #     i = a.send(itr)
    #     return i

    # a = abyss()
    
    # def fn():
    #     x = choi(range(10), a)
    #     y = choi(range(10), a)
    #     z = choi(range(10), a)
    #     w = choi(range(10), a)
    #     v = choi(range(10), a)
    #     return x, y, z, w, v

    def abyss(itr=(None,), choices=()):
        for i, item in enumerate(itr):
            if i > 0:
                for c in choices: yield c
            result = (yield item)
            if result != None:
                yield from abyss(iter(result), choices + (item,))

    class NonDeterminator:
        def __init__(self):
            self.abyss = None # or could default to random sampler...

        def choice(self, itr):
            return self.abyss.send(itr)

        def run(self, f):
            self.abyss = abyss()
            for _ in self.abyss: yield f()

    nd = NonDeterminator()
    def fn():
        x = nd.choice(range(10))
        y = nd.choice(range(10))
        z = nd.choice(range(10))
        w = nd.choice(range(10))
        v = nd.choice(range(10))
        return x, y, z, w, v

    from time import perf_counter

    start = perf_counter()
    # run(fn, a)
    for ret in nd.run(fn):
        # print(ret)
        res = ret
    abyss_stop = perf_counter() - start

    # works, just ugly impl of Runner
    r = Runner()
    def fn():
        x = r.choice(range(10))
        y = r.choice(range(10))
        z = r.choice(range(10))
        w = r.choice(range(10))
        v = r.choice(range(10))
        return x, y, z, w, v

    start = perf_counter()    
    for ret in r.run(fn):
        # print(ret)
        res = ret
    try_stop = perf_counter() - start
    
    # comparable runtimes
    print("abyss", abyss_stop)
    print("try", try_stop)

