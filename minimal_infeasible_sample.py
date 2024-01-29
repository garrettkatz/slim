import sys, os
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
import scipy.sparse as sp
from check_span_rule import *

do_exp = True
do_show = True

solver = sys.argv[1]
N = int(sys.argv[2])

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# sorter = np.argsort(-B.sum(axis=1))
sorter = np.argsort(B.sum(axis=1)) # sequential infeasible at sample [0, 1, ..., 503]
Y = Y[sorter]
B = B[sorter]
W = W[sorter]

print('infeas', sorter[215], sorter[503])

if do_exp:

    found_infeasible = False

    # # sequential
    # for num_regions in range(2, len(Y)):
    #     sample = np.arange(num_regions)

    #     print(f"region sample of {len(Y)}", sample)
    #     result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=False)
    #     status, u, g, D, E = result

    #     if status != "optimal":
    #         found_infeasible = True
    #         fname = f"minimal_infeasible_sequential_{solver}_{N}.pkl"
    #         with open(fname, 'wb') as f: pk.dump((result, sample), f)

    #     if found_infeasible: break

    # combos
    hi = 504 # from sequential
    for num_regions in range(2, hi):
        for sample in map(list, it.combinations(range(hi), num_regions)):

            # sample = np.arange(hi) # what sequential found
            sample = [215, 503] # what combos found.  GUROBI and SCIPY agree, even with reoptimize=True

            # sample = np.random.choice(np.arange(504, len(Y)), 500, replace=False)
            # sample[:2] = [215, 503]

            print(f"region sample of {len(Y)}", sample)
            result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=True)
            status, u, g, D, E = result

            if status != "optimal":
                print(f"status={status}")
                found_infeasible = True
                fname = f"minimal_infeasible_combos_{solver}_{N}.pkl"
                with open(fname, 'wb') as f: pk.dump((result, sample), f)

            if found_infeasible: break
        if found_infeasible: break

    if not found_infeasible:
        print("All sub-samples feasible")

if do_show:

    fname = f"minimal_infeasible_combos_{solver}_{N}.pkl"
    if not os.path.exists(fname):
        print("All sub-samples feasible")
        sys.exit(1)

    with open(fname, 'rb') as f: result, sample = pk.load(f)

    print("sample:", sample)
    print("sorter[sample]:", sorter[sample])

    Bu = B[sample].any(axis=0)

    ### zeroing in on submatrices
    B = B[sample][:, Bu]
    XT = X.T[:, Bu]
    Y = Y[sample][:, Bu]
    W = W[sample]

    print("B:")
    print(B.astype(int))
    print("X.T:")
    print(XT)
    print("Y:")
    print(Y)
    print("W:")
    print(W)

    print("No shared X in separate branches, per-X Y:")
    y = Y[0].copy()
    y[B[1]] = Y[1][B[1]]
    print(y)

    print("xy for w(xy) constraints:")
    xy = XT*y
    print(xy)

    print("block for 'w0' after shared leading path:")
    idx = np.flatnonzero(B[0] == B[1])
    xy0 = xy[:,idx]
    print('B idx', idx)
    print(xy0)

    print("block for w01, w12 in top branch")
    idx = np.flatnonzero(B[0] & ~B[1])
    xy_top = xy[:,idx]
    print('B idx', idx)
    print(xy_top)

    print("block for w03, w34, w45 in bottom branch")
    idx = np.flatnonzero(~B[0] & B[1])
    xy_bot = xy[:,idx]
    print('B idx', idx)
    print(xy_bot)

    print("dataset for w01: w0 xy01 + g01 x01 xy01")
    xy01 = np.append(xy0, xy_top[:,:1], axis=1)
    xy01 = np.append(xy01, xy_top[:,:1].T @ xy01, axis=0)
    print(xy01)

    print("dataset for w12: w0 xy012 + g01 x01 xy012 + g12 x12 xy012")
    xy012 = np.append(xy0, xy_top[:,:2], axis=1)
    xy012 = np.append(xy012, xy_top[:,:2].T @ xy012, axis=0)
    print(xy012)

    print("dataset for w03: w0 xy03 + g03 x03 xy03")
    xy03 = np.append(xy0, xy_bot[:,:1], axis=1)
    xy03 = np.append(xy03, xy_bot[:,:1].T @ xy03, axis=0)
    print(xy03)

    print("dataset for w34: w0 xy034 + g03 x03 xy034 + g34 x34 xy034")
    xy034 = np.append(xy0, xy_bot[:,:2], axis=1)
    xy034 = np.append(xy034, xy_bot[:,:2].T @ xy034, axis=0)
    print(xy034)

    print("dataset for w45: w0 xy0345 + g03 x03 xy0345 + g34 x34 xy0345 + g45 x45 xy0345")
    xy0345 = np.append(xy0, xy_bot[:,:3], axis=1)
    xy0345 = np.append(xy0345, xy_bot[:,:3].T @ xy0345, axis=0)
    print(xy0345)

    print("full coefficient matrix A:")

    A = sp.bmat([
        [xy0,  xy01[0:8], xy012[0:8 ],  xy03[0:8], xy034[0:8],  xy0345[0 :8 ]],
        [None, xy01[8:9], xy012[8:9 ],  None,      None,        None         ],
        [None, None,      xy012[9:10],  None,      None,        None         ],
        [None, None,      None,         xy03[8:9], xy034[8:9],  xy0345[8 :9 ]],
        [None, None,      None,         None,      xy034[9:10], xy0345[9 :10]],
        [None, None,      None,         None,      None,        xy0345[10:11]],
    ]).toarray().astype(int).T

    # halfsies
    np.set_printoptions(linewidth=200)

    # premute vars
    A = A[:,::-1]

    # augment bias
    A = np.block([A, np.ones((len(A), 1), dtype=int)])

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,1)
    pt.imshow(A)

    # eliminating one of two signs in each column
    sgn = -1
    i = np.argmax(np.sign(A[:,0]) == sgn)
    opp = (np.sign(A[:,0]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[0]]) / np.fabs(A[i,0])
    A = A[np.sign(A[:,0]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,2)
    pt.imshow(A)

    # next column has fewer negatives, cancel + keep positive rows
    sgn = -1
    i = np.argmax(np.sign(A[:,1]) == sgn)
    opp = (np.sign(A[:,1]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[1]]) / np.fabs(A[i,1])
    A = A[np.sign(A[:,1]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,3)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows
    sgn = +1
    i = np.argmax(np.sign(A[:,2]) == sgn)
    opp = (np.sign(A[:,2]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[2]]) / np.fabs(A[i,2])
    A = A[np.sign(A[:,2]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,4)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows
    sgn = +1
    i = np.argmax(np.sign(A[:,3]) == sgn)
    opp = (np.sign(A[:,3]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[3]]) / np.fabs(A[i,3])
    A = A[np.sign(A[:,3]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,5)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows
    sgn = +1
    i = np.argmax(np.sign(A[:,4]) == sgn)
    opp = (np.sign(A[:,4]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[4]]) / np.fabs(A[i,4])
    A = A[np.sign(A[:,4]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,6)
    pt.imshow(A)

    # next column has fewer negatives, cancel + keep positive rows
    sgn = -1
    i = np.argmax(np.sign(A[:,5]) == sgn)
    opp = (np.sign(A[:,5]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[5]]) / np.fabs(A[i,5])
    A = A[np.sign(A[:,5]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,7)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows
    sgn = +1
    i = np.argmax(np.sign(A[:,6]) == sgn)
    opp = (np.sign(A[:,6]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[6]]) / np.fabs(A[i,6])
    A = A[np.sign(A[:,6]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,8)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows (well, equal this time)
    sgn = +1
    i = np.argmax(np.sign(A[:,7]) == sgn)
    opp = (np.sign(A[:,7]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[7]]) / np.fabs(A[i,7])
    A = A[np.sign(A[:,7]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,9)
    pt.imshow(A)

    # next column has fewer negatives, cancel + keep positive rows
    sgn = -1
    i = np.argmax(np.sign(A[:,8]) == sgn)
    opp = (np.sign(A[:,8]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[8]]) / np.fabs(A[i,8])
    A = A[np.sign(A[:,8]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,10)
    pt.imshow(A)

    # next column has fewer negatives, cancel + keep positive rows
    sgn = -1
    i = np.argmax(np.sign(A[:,9]) == sgn)
    opp = (np.sign(A[:,9]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[9]]) / np.fabs(A[i,9])
    A = A[np.sign(A[:,9]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,11)
    pt.imshow(A)

    # next column has more negatives, cancel + keep negative rows (equal this time, but choice important to also keep some zero rows)
    sgn = +1
    i = np.argmax(np.sign(A[:,10]) == sgn)
    opp = (np.sign(A[:,10]) == -sgn)
    A[opp] = A[opp] + A[i] * np.fabs(A[opp][:,[10]]) / np.fabs(A[i,10])
    A = A[np.sign(A[:,10]) != sgn]

    print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    pt.subplot(1,13,12)
    pt.imshow(A)

    # fifth column all positive, no zeros even, stuck

    # # augment for slack variables
    # AIb = np.concatenate((A, -np.eye(len(A)), np.ones((len(A),1))), axis=1)

    # # row-reduce AIb manually, first 8 weights
    # i = -1
    # for j in range(A.shape[1]):
    #     i = np.argmax(AIb[i+1:,j] != 0) + (i+1)
    #     print(i,j)
    #     AIb[i] /= AIb[i,j]
    #     for k in range(len(AIb)):
    #         if k == i: continue
    #         AIb[k] -= AIb[i]*(AIb[k,j]/AIb[i,j])

    # pt.figure()
    # pt.imshow(AIb)

    # # row-reduce A positive row scaling and addition
    # np.set_printoptions(linewidth=200)
    # A = np.block([A, np.ones((len(A), 1), dtype=int)])

    # pt.subplot(1,11,1)
    # pt.imshow(A)

    # # A[1,0] < 0, add to all lower positive rows
    # lowpos = A[:,0] > 0
    # lowpos[:2] = False
    # A[lowpos] = A[lowpos] + A[1]
    # # A[0,0] > 0, add to all lower negative rows
    # lowneg = A[:,0] < 0
    # lowneg[:1] = False
    # A[lowneg] = A[lowneg] + A[0]

    # # halve remaining rows, should be all even.  only happens for A[0,0]?
    # assert (A[1:] % 2 == 0).all()
    # A[1:] = A[1:]//2

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,2)
    # pt.imshow(A)

    # # now permute a positive 1 into row 1
    # A[[1, 5]] = A[[5, 1]]
    # # A[3,1] < 0, add to all positive rows other than 1
    # pos = A[:,1] > 0
    # pos[1] = False
    # A[pos] = A[pos] + A[3]
    # # A[1,1] > 0, add to all other negative rows
    # neg = A[:,1] < 0
    # neg[1] = False
    # A[neg] = A[neg] + A[1]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,3)
    # pt.imshow(A)

    # # now permute a positive 1 in row 2
    # A[[2, 12]] = A[[12, 2]]
    # # A[20,2] < 0, add to all positive rows other than 2
    # pos = A[:,2] > 0
    # pos[2] = False
    # A[pos] = A[pos] + A[20]
    # # A[2,2] > 0, add to all other negative rows
    # neg = A[:,2] < 0
    # neg[2] = False
    # A[neg] = A[neg] + A[2]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,4)
    # pt.imshow(A)

    # # now col 3 has several +1's (incl [3,3]), 0's and no -1's. sort the +1's up
    # sorter = np.argsort(-A[4:,3], kind='stable')
    # A[4:] = A[4:][sorter]
    # print('sorter',sorter)
    
    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,5)
    # pt.imshow(A)

    # # A[27,4] < 0, add to all positive rows other than 16
    # pos = A[:,4] > 0
    # pos[16] = False
    # A[pos] = A[pos] + A[27]
    # # A[16,4] > 0, add to all other negative rows
    # neg = A[:,4] < 0
    # neg[16] = False
    # A[neg] = A[neg] + A[16]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,6)
    # pt.imshow(A)

    # # in col 5, all rows 17 and below are now +1 or +2, and above are > 0, so you can't cancel anything in col 5
    # # just sort the ones below to be orderly

    # sorter = np.argsort(-A[17:,5], kind='stable')
    # A[17:] = A[17:][sorter]
    # print('sorter',sorter)
    
    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,7)
    # pt.imshow(A)

    # # col 6 is all non-positive, can't cancel any
    # assert (A[:,6] <= 0).all()

    # # A[28,7] < 0, add to all positive rows other than 18
    # pos = A[:,7] > 0
    # pos[18] = False
    # A[pos] = A[pos] + A[28]
    # # A[18,7] > 0, add to all other negative rows
    # neg = A[:,7] < 0
    # neg[18] = False
    # A[neg] = A[neg] + A[18]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,8)
    # pt.imshow(A)

    # # col 8 is all pos below, but sort again to stay orderly
    # sorter = np.argsort(-A[19:,8], kind='stable')
    # A[19:] = A[19:][sorter]
    # print('sorter',sorter)

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,9)
    # pt.imshow(A)

    # # col 9 has a +1 and -1 below, first permute +1 to be orderly
    # A[[34, 37]] = A[[37,34]]
    # # use -1 in [38,9] to cancel all positives except 34 where cols 5,6 are also nonzero
    # posnz = (A[:,[5,6]] != 0).all(axis=1) & (A[:,9] > 0)
    # posnz[34] = False
    # assert (A[posnz][:,[9]] > 0).all() # make sure you don't flip inequalities
    # A[posnz] = A[posnz] + A[38] * A[posnz][:,[9]]

    # # then cancel 38 with 34 (two more -1's but not in 5,6 nz rows)
    # A[38] = A[38] + A[34]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,10)
    # pt.imshow(A)

    # # col 10 mostly negative, sort to be orderly
    # sorter = np.argsort(A[35:,10], kind='stable')
    # A[35:] = A[35:][sorter]
    # print('sorter',sorter)

    # # then use 35 to cancel a few +4's in nz rows above
    # A[31:34] = A[31:34] + 4*A[35]

    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))
    # pt.subplot(1,11,11)
    # pt.imshow(A)

    # # looks like you're done with echelon, bottom right all zero now
    
    # # remove a few duplicates
    # _, idx = np.unique(A, return_index=True, axis=0)
    # idx = np.sort(idx)
    # A = A[idx]
    # print(np.block([[0, np.arange(A.shape[1])], [np.arange(A.shape[0]).reshape(-1,1), A]]))

    # print("LP simplex canonical tableau with artificial variables:")
    # # [w,...g...] = v = (v^+) -  (v^-)
    # # v+, v- >= 0
    # tab = sp.bmat([
    #     [1, None, None, None, -np.ones((1,len(A))), None],
    #     [None, A, -A, -np.eye(len(A)), np.eye(len(A)), np.ones((len(A),1))]
    # ]).toarray()

    # print(f"confirming tab has full rank: {np.linalg.matrix_rank(tab)} == {len(tab)}")

    # # price out artificial variables
    # tab[0] += 2*tab[1:].sum(axis=0)

    # print(tab[:,:A.shape[1]])

    
    # pt.subplot(3,1,1)
    # pt.imshow(B)
    # pt.title("Boundaries")
    # pt.ylabel("i")
    # # pt.axis("off")

    # pt.subplot(3,1,2)
    # pt.imshow(XT)
    # pt.title("Vertices")
    # # pt.axis("off")

    # pt.subplot(3,1,3)
    # pt.imshow(Y)
    # pt.ylabel("i")
    # pt.xlabel("k")
    # pt.title("Dichotomies")
    # # pt.axis("off")

    # pt.figure()
    # pt.subplot(1,3,1)
    # pt.imshow(xy.T)
    # pt.subplot(1,3,2)
    # pt.imshow(A)
    # pt.subplot(1,3,3)
    # pt.imshow(tab)
    # pt.colorbar()

    pt.show()

