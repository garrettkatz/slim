"""
This might all be doable more elegantly using coroutines/yield from:
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

    r = Runner()
    def fn():
        x = r.choice(range(3))
        if x < 2:
            y = r.choice(range(3))
        else:
            # y = r.choice(range(4))
            y = r.choice([])
        return x,y

    for ret in r.run(fn):
        print(ret)
        # if x == 3: break
    
    
