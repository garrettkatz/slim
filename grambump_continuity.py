import pickle as pk
from geneng import load_data
from grambump import *

variables = (Variable("w", Vector), Variable("x", Vector), Variable("y", Scalar), Variable("N", Scalar))
w, x, y, N = variables

# all patterns in a group must contain the same parameters
P = tuple(Parameter(n) for n in range(3))

# classical, any op can be replaced by another of the same signature
pointwise = (
    (w, x, ),
    (y, N, Constant(-1), Constant(0), Constant(1), ),
    (Sign(P[0]), -P[0], Inv(P[0]), Abs(P[0]), Sqrt(P[0]), Log(P[0]), Exp(P[0]), ),
    (Sum(P[0]), Prod(P[0]), Mean(P[0]), Least(P[0]), Largest(P[0]), ), 
    (P[0] + P[1], P[0] * P[1], Max(P[0], P[1]), Min(P[0], P[1]), ),
    (Dot(P[0], P[1]), WhereLargest(P[0], P[1]), ),
)

liberal = (
    # terminals
    (w, Sign(w), ),
    # (w, x, y * w, y * x, w * x, Sign(w), ),
    (y, Prod(x), Sign(Sum(x)), Least(x), Largest(x), Sign(Sum(w * x)), Constant(1), Constant(-1)),
    (N, Abs(y), Constant(3), Constant(4), Sum(x), Largest(w), ),

    # unaries
    (Sign(P[0]), Sqrt(P[0]) * Sign(P[0]), Log(Abs(P[0]) + 1) * Sign(P[0]), ),
    (Neg(P[0]), Exp(Neg(P[0])) - 1, ),
    (Inv(P[0]), Sign(P[0]) * Exp(Abs(Neg(P[0]) - 1)), Sign(P[0]) * Max(Neg(Abs(P[0])) + 2, Constant(0)), ),
    # (Square(P[0]), Abs(P[0]), ), # omit redundancies
    (Abs(P[0]), P[0] * P[0], ),
    (Sqrt(P[0]), Abs(P[0]), ),
    (Log(P[0]), Sqrt(P[0]) - 1, ),
    (Exp(P[0]), Max(P[0] + 1, Constant(0)), Max(Sign(P[0] + 1) * (P[0] + 1) * (P[0] + 1), Constant(0)), ),
    (Sum(P[0]), Sum(P[0] * y), Sum(P[0] * x), (Least(P[0]) + Largest(P[0])) * Inv(Constant(2)) * N, ),
    (Prod(P[0]), Prod(Sign(P[0])), ),
    # (Mean(P[0]), (Least(P[0]) + Largest(P[0])) * Inv(Constant(2)), ), # omit redundancies
    (Least(P[0]), Least(Sign(P[0])), ),
    (Largest(P[0]), Largest(Sign(P[0])), ),

    # binaries
    (P[0] + P[1], Max(P[0], P[1]) * 2, Min(P[0], P[1]) * 2, ),
    (P[0] * P[1], Min(P[0] * P[0], P[1] * P[1]) * Sign(P[0] * P[1]), ),
    (P[0] * P[0], Exp(Abs(P[0])) - 1, ),
    (Min(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
    (Max(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
    # (Dot(P[0], P[1]), Least(P[0] * P[1]) + Largest(P[0] * P[1]), ), # leave out until simplifications

    # # open-ended
    # (P[0], P[0] + y, P[0] + x, P[0] + 1, P[0] - 1, ),

)

# conservative: should very small deviation
conservative = (

    # terminals
    # (w, Max(w, x), Max(w, y), Abs(x) * Mean(w), ),
    (x, ),
    # (y, Min(y, Sum(w)), ),
    (N, Constant(3), Constant(4), ),

    # # unaries
    (Sign(P[0]), Min(1, Sqrt(P[0])) * Sign(P[0]), ),
    (Inv(P[0]), ),
    # (Inv(P[0]), Max(Inv(P[0]), Sign(P[0]) * Exp(-Abs(P[0] - 1))), ), #Sign(P[0]) * Max(-Abs(P[0]) + 2, 0), ),
    (Square(P[0]), Min(Square(P[0]), Abs(Square(P[0]) * P[0])), ),
    (Abs(P[0]), P[0] * Sign(P[0]), ),
    # (Abs(P[0]), Max(Abs(P[0]), Sqrt(P[0])), Min(Abs(P[0]), Square(P[0])), ),
    (Sqrt(P[0]), ), #Min(Sqrt(P[0]), Abs(P[0])), ),
    # (Log(P[0]), Sqrt(P[0]) - 1, ),
    # (Exp(P[0]), Max(P[0] + 1, Constant(0)), Max(Sign(P[0] + 1) * (P[0] + 1) * (P[0] + 1), Constant(0)), ),
    (Sum(P[0]), (Least(P[0]) + Largest(P[0])) * N / 2, ),
    # (Sum(P[0]), Sum(P[0] * y), Sum(P[0] * x), (Least(P[0]) + Largest(P[0])) * Inv(Constant(2)) * N, ),
    # (Prod(P[0]), Prod(Sign(P[0])), ),
    (Mean(P[0]), Sum(P[0]) / N, (Least(P[0]) + Largest(P[0])) / 2, ), # omit redundancies
    # (Least(P[0]), Least(Sign(P[0])), ),
    # (Largest(P[0]), Largest(Sign(P[0])), ),

    # not quite identities since exp/log are truncated
    (Sum(P[0]), Log(Prod(Exp(P[0]))), ),
    (Prod(P[0]), Exp(Sum(Log(P[0]))), ),
    (Log(P[0] * P[1]), Log(P[0]) + Log(P[1]), ),
    (Exp(P[0] + P[1]), Exp(P[0]) * Exp(P[1]), ),

    # binaries
    (P[0] + P[1], ), # Max(P[0], P[1]) * 2, Min(P[0], P[1]) * 2, ),
    (P[0] * P[1], Min(Square(P[0]), Square(P[1])) * Sign(P[0] * P[1]), ),
    # (P[0] * P[0], Exp(Abs(P[0])) - 1, ),
    # (Min(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
    (Min(P[0], P[1]), Sqrt(P[0] * P[1]) * Sign(P[0] * P[1])),
    # (Max(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
    # (Dot(P[0], P[1]), Least(P[0] * P[1]) + Largest(P[0] * P[1]), ), # leave out until simplifications

    # wheres
    (WhereLargest(P[0], P[1]), ),
    (WherePositive(P[0], P[1], P[2]), ((Sign(P[0]) + 1) / 2) * P[1] + ((Sign(-P[0]) + 1) / 2) * P[2], ),

    # # open-ended
    # (P[0], P[0] + (y / N), P[0] + (x / N), P[0] + Inv(N), P[0] - Inv(N), P[0] + (Exp(-Abs(x)) / N),  P[0] + (Inv(Abs(x) + 1) / N), ),

)

# should be zero deviation
identities = (
    (w, Abs(x)*w, ),
    (x, Abs(x)*x, ),
    (y, Sign(N)*y, ),
    (Abs(y), Constant(1), ),
    (N, Abs(y)*N, ),
    (Sum(P[0]), Mean(P[0]) * N, ),
    (P[0], Abs(y)*P[0], Sign(N)*P[0], Sign(P[0])*Sqrt(Square(P[0])), ),
    (P[0]*P[0], Square(P[0]), ),
    (P[0]*(P[1] + P[2]), (P[0]*P[1]) + (P[0]*P[2]), ),
    (Dot(P[0], P[1]), Sum(P[0] * P[1]), ),
    (Least(P[0]), WhereLargest(-P[0], P[0]), ),
    (Largest(P[0]), WhereLargest(P[0], P[0]), ),
    (Max(P[0], P[1]), -Min(-P[0], -P[1]), ),
)

if __name__ == "__main__":

    dataset = load_data(Ns=[3,4], perceptron=False)
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
    
    constants = tuple(Constant(v, Scalar) for v in range(-1,2))
    operators = { # out_type: classes
        Output: (Sign, Abs, Neg, Inv, Sqrt, Square, Log, Exp, Add, Mul, Min, Max, WherePositive),
        Scalar: (Sum, Prod, Least, Largest, Mean, Dot, WhereLargest),
    }

    neighborhood_relations = (liberal, conservative, identities)
    results = []
    for n, neighborhoods in enumerate(neighborhood_relations):
    
        guide = Guide(constants + variables, operators, neighborhoods)
        
        # print(guide.neighbors(x))
        # input('.')
        
        results.append([])
        reps = 200
        for rep in range(reps):
        
            formula = guide.sprout(SpanRule, Vector, term_prob=.5, max_depth=8)
            base_fit = fitness_function(formula)
        
            for neighbor in guide.neighbors(formula):
                neighbor_fit = fitness_function(neighbor)
                results[n].append((base_fit, neighbor_fit - base_fit))
        
        
    import matplotlib.pyplot as pt
    pt.figure(figsize=(8, 4))
    num_cols = len(neighborhood_relations)
    for n, neighborhoods in enumerate(neighborhood_relations):
        fits, bumps = zip(*results[n])
        pt.subplot(2,num_cols,n+1)
        pt.hist(bumps, bins=np.linspace(-1, 1, 20))
        pt.xlabel("Fitness change")
        pt.ylabel("Frequency")
        pt.title(["liberal", "conservative", "identities"][n])
        pt.subplot(2,num_cols,num_cols+n+1)
        pt.plot(fits, bumps, 'k.')
        pt.xlabel("Base fitness")
        pt.ylabel("Fitness change")
    pt.tight_layout()
    pt.savefig('grambump_cty.png')
    pt.show()
        
        
