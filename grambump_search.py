from grambump import *
import grambump_continuity as gbc

def greedy(guide, form, fit_fun, max_depth, max_itrs):
    fitness = fit_fun(form)
    explored = {}
    for itr in range(max_itrs):

        improved = False
        for neighbor in guide.neighbors(form):
            ns = str(neighbor)
            if ns in explored: continue

            nf = fit_fun(neighbor)
            explored[ns] = nf

            if neighbor.depth() > max_depth: continue

            if nf > fitness:
                form = neighbor
                fitness = nf
                improved = True

        if not improved: break

    return form, fitness, itr+1, explored

from heapq import heappop, heappush

def queued(guide, form, fit_fun, max_depth, max_itrs, max_queue, fit_target=.9999):
    max_form, max_fit = form, fit_fun(form)
    queue = [(-max_fit, 0, max_form)] # min heap
    push_order = 1 # break ties with FIFO
    explored = {}
    for itr in range(max_itrs):
        # stop if queue is exhausted
        if len(queue) == 0: break

        # pop best formula so far
        nfi, _, form = heappop(queue)

        # check its neighbors
        for neighbor in guide.neighbors(form):

            # skip neighbor if already explored/pushed
            ns = str(neighbor)
            if ns in explored: continue

            # not explored yet, evaluate
            nf = fit_fun(neighbor)
            explored[ns] = nf

            # print(f"   {itr} :: {len(queue)} :: {len(explored)} :: {max_fit:.6f} vs {nf:.6f} {neighbor}")

            # track best found so far
            if nf > max_fit: max_form, max_fit = neighbor, nf

            # break if target reached
            if max_fit  > fit_target: break

            # don't push formulas over the depth limit
            if neighbor.depth() > max_depth: continue

            # clean queue if over the size limit (remove lowest entries)
            if len(queue) >= max_queue:
                queue = queue[:len(queue) // 2]

            # push new node, track order for FIFO
            heappush(queue, (-nf, push_order, neighbor)) # min heap
            push_order += 1

    return max_form, max_fit, itr+1, explored

class CappedFrontier:
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = []
        self.count = 0

    def __len__(self):
        return len(self.heap)

    def peek(self):
        neg_fit, _, form = self.heap[0]
        return form, -neg_fit

    def pop(self):
        neg_fit, _, form = heappop(self.heap)
        return form, -neg_fit

    def push(self, form, fit):
        if len(self.heap) == self.capacity:
            self.heap.pop()
        heappush(self.heap, (-fit, self.count, form))
        self.count += 1

class CappedExplored:
    def __init__(self, capacity):
        self.capacity = capacity
        self.lookup = {}
        self.heap = []
        self.count = 0

    def __len__(self):
        return len(self.lookup)

    def __contains__(self, form_str):
        return form_str in self.lookup

    def __getitem__(self, form_str):
        return self.lookup[form_str]

    def __setitem__(self, form_str, fit):
        if len(self.heap) == self.capacity:
            old_fit, _, old_str = heappop(self.heap)
            del self.lookup[old_str]
        self.lookup[form_str] = fit
        heappush(self.heap, (fit, self.count, form_str))
        self.count += 1

def screened_push(form, fit_fun, explored, frontier):
    fs = str(form)
    if fs in explored: return True, explored[fs]
    explored[fs] = fit = fit_fun(form)
    frontier.push(form, fit)
    return False, fit

def multisprout(guide, fit_fun, max_depth, capacity, base_name, max_itrs = None, fit_target=.9999):

    max_form = None
    max_fit = -1
    num_itrs = 0
    num_evals = 0
    num_sprouts = 0
    avg_greedy = 0
    results = []

    frontier = CappedFrontier(capacity)
    explored = CappedExplored(capacity)

    while max_fit < fit_target and num_itrs != max_itrs:
        num_itrs += 1

        if len(frontier) == 0 or frontier.peek()[1] <= avg_greedy:

            form = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,max_depth+1))
            already_explored, fit = screened_push(form, fit_fun, explored, frontier)
            if not already_explored:
                num_evals += 1

                # greedy local max
                while True:
    
                    improved = False
                    for neighbor in guide.neighbors(form):
                        already_explored, nfit = screened_push(neighbor, fit_fun, explored, frontier)
                        if already_explored: continue
                        num_evals += 1
    
                        if nfit > fit:
                            improved = True
                            form, fit = neighbor, nfit
                            print(f"   {num_itrs} :: {len(frontier)} :: {len(explored)} :: greedy :: {max_fit:.6f} vs {fit:.6f} {form}")

                    if not improved: break

            # update running average
            num_sprouts += 1
            avg_greedy += (fit - avg_greedy) / num_sprouts

        else:    

            form, fit = frontier.pop()

            # update fitness
            if fit > max_fit:
                max_form, max_fit = form, fit
                results.append((num_sprouts, num_evals, max_fit, max_form))
                with open(base_name, "wb") as f: pk.dump(results, f)

            print(f"   {num_itrs} :: {len(frontier)} :: {len(explored)} :: avg {avg_greedy} :: {max_fit:.6f} vs {fit:.6f} {form}")

            for neighbor in guide.neighbors(form):
                already_explored, nfit = screened_push(neighbor, fit_fun, explored, frontier)
                if not already_explored: num_evals += 1

    return results

if __name__ == "__main__":

    import sys
    tag = sys.argv[1] # used for saving results

    import pickle as pk
    from geneng import load_data

    do_perceptron = True
    do_spanrule = False
    max_depth = 6 if do_perceptron and do_spanrule else 10

    do_random = False
    do_greedy = True
    do_queued = False
    do_multi = False
    do_show = False

    # max_evals = 500_000
    max_evals = 1_000_000_000

    dataset = load_data(Ns=[3,4], perceptron=do_perceptron)
    # dataset[n]: [w_old, x, y, w_new, margins]

    dataset = [
        ({"w": dat[0], "x": dat[1], "y": dat[2], "N": np.array([[float(dat[0].shape[1])]])}, dat[3])
        for dat in dataset]

    print("num examples:")
    print(dataset[0][0]["y"].shape, dataset[1][0]["y"].shape)

    def fitness_function(formula):
        fitness = 0.
        for inputs, w_new in dataset:
            # invoke formula
            w_pred = formula(inputs)

            # broadcast scalar outputs
            if w_pred.shape[1] == 1: w_pred = w_pred * np.ones(w_new.shape[1])

            # mean cosine similarity
            w_pred = w_pred / np.maximum(np.linalg.norm(w_pred, axis=1, keepdims=True), 10e-8)
            fitness += (w_pred * w_new).sum(axis=1).mean() # cosine similarity

        return fitness / len(dataset)

    variables = (Variable("w", Vector), Variable("x", Vector), Variable("y", Scalar), Variable("N", Scalar))
    w, x, y, N = variables

    if do_perceptron:
        constants = tuple(Constant(v, Scalar) for v in range(1,2))
        operators = {
            Output: (Sign, Neg, Add, Mul),
            Scalar: (Sum,),
        }
    else:
        constants = tuple(Constant(v, Scalar) for v in range(-1,2))
        operators = { # out_type: classes
            Output: (Sign, Abs, Neg, Inv, Sqrt, Square, Log, Exp, Add, Mul, Min, Max, WherePositive),
            Scalar: (Sum, Prod, Least, Largest, Mean, Dot, WhereLargest),
        }

    # guide = Guide(constants + variables, operators, neighborhoods=gbc.identities + gbc.conservative)
    guide = Guide(constants + variables, operators, neighborhoods=gbc.liberal)
    # guide = Guide(constants + variables, operators, neighborhoods=gbc.conservative)
    # guide = Guide(constants + variables, operators, neighborhoods=gbc.pointwise)

    if do_random:

        print("\n********************** random sampling\n")
        max_fit = -1
        max_span = None
        for rep in range(max_evals):
            span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=6)
            fit = fitness_function(span)
            if fit > max_fit:
                max_fit, max_span = fit, span
                print(f"{rep}: {fit} vs {max_fit} <- {max_span}")
                # print(max_span.tree_str())
                if max_fit > .99999: break
            # print(f"{rep}: {fit} vs {max_fit} <- {span}, {max_span}")

    if do_greedy:    

        print("\n********************** repeated greedy\n")
        max_fit = -1
        max_form = None
        num_evals = 0
        num_sprouts = 0
        results = []
        while num_evals < max_evals:
            num_sprouts += 1

            op_cls = SpanRule if do_spanrule else np.random.choice(operators[Output])
            form = guide.sprout(op_cls, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,max_depth+1))
            form, fit, num_itrs, explored = greedy(guide, form, fitness_function, max_depth=max_depth, max_itrs=500)

            # # check that final performance always better for queued, greedy is a special case
            # formq, fitq, num_itrsq, exploredq = queued(guide, form, fitness_function, max_depth=max_depth, max_itrs=30, max_queue=500)
            # print(f"{fit:.4f} vs {fitq:.4f}, {num_itrs} vs {num_itrsq}, {len(explored)} vs {len(exploredq)}")

            num_evals += len(explored)
            if fit > max_fit:
                form = form.lump_constants()
                max_fit, max_form = fit, form
                # print(f"{num_sprouts} sprouts {num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit:.4f} <- {max_form}")
                # print(max_form.tree_str())
                results.append((num_sprouts, num_evals, max_fit, max_form))
                with open(f"grambump_greedy_{tag}.pkl", "wb") as f: pk.dump(results, f)
            print(f"{num_sprouts} sprouts {num_evals} evals ({num_itrs:>2d} itrs|{len(explored):>3d} eval'd): {max_fit:.4f} ~ {max_form} vs {fit:.4f}")
            if max_fit > .99999: break

    if do_queued:
    
        print("\n********************** repeated queued\n")
        max_fit = -1
        max_span = None
        num_evals = 0
        num_sprouts = 0
        results = []
        while num_evals < max_evals:
            num_sprouts += 1
            if do_perceptron:
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,6+1))
                span, fit, num_itrs, explored = queued(guide, span, fitness_function, max_depth=6, max_itrs=1000, max_queue=2**12)
            else:
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,8+1))
                span, fit, num_itrs, explored = queued(guide, span, fitness_function, max_depth=8, max_itrs=1000, max_queue=2000)
            print(f"{num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit:.4f} vs {fit:.4f} <- {span}")
            num_evals += len(explored)
            if fit > max_fit:
                max_fit, max_span = fit, span
                print(f"{num_sprouts} sprouts {num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit} <- {max_span}")
                # print(max_span.tree_str())
                results.append((num_sprouts, num_evals, max_fit, max_span))
                with open(f"grambump_queued_{tag}.pkl", "wb") as f: pk.dump(results, f)
                if max_fit > .99999: break

    if do_multi:
        if do_perceptron:
            results = multisprout(guide, fitness_function, max_depth=8, capacity=2**12, base_name=f"grambump_multi_{tag}.pkl", fit_target=.9999, max_itrs=100)
        else:
            results = multisprout(guide, fitness_function, max_depth=6, capacity=2**12, base_name=f"grambump_multi_{tag}.pkl", fit_target=.9999, max_itrs=100)

    if do_show:
        import matplotlib.pyplot as pt

        results = {}
        for key in ("greedy", "queued"):
            try:
                with open(f"grambump_{key}_{tag}.pkl", "rb") as f: results[key] = pk.load(f)
            except:
                pass # didn't run that experiment yet

        pt.figure()
        for key in results:
            print(key)
            num_sprouts, num_evals, max_fit, max_span = zip(*results[key])
            pt.plot(num_evals, max_fit, 'o-', label=key)
            print(f"best fit {key}: {max_span}")
        pt.legend()
        pt.show()
