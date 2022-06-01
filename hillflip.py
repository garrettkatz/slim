import matplotlib.pyplot as pt
import itertools as it
import numpy as np

N = 6
u = np.random.randn(N)
v = np.random.randn(N)

X = np.array(tuple(it.product((-1, 1), repeat=N)))

alp = -X.dot(u) / X.dot(v)
pos = np.flatnonzero(alp > 0)
opt = pos[alp[pos].argmin()]

print(opt)

i = {0: np.random.randint(2**N)}
x = X[i[0]]
for n in it.count():
    a = - x.dot(u) / x.dot(v)
    da = - (x.dot(u) - 2*x*u) / (x.dot(v) - 2*x*v)
    daa = np.concatenate((da, [a]))
    print(f'iter {n}: a={a}, x, daa:')
    print(x)
    print(daa)
    
    if (daa <= 0).all():
        flip = daa.argmax()
        print(f"n={n}: all neg, flip {flip}")
    else:
        pos = np.flatnonzero(daa > 0)
        flip = pos[daa[pos].argmin()]
        print(f"n={n}: some pos. pos, daa[pos], argmin:")
        print(f"{pos}")
        print(f"{daa[pos]}")
        print(f"{pos[daa[pos].argmin()]}")
    
    if flip == N:
        print("local opt")
        break

    x = x.copy()
    x[flip] *= -1
    i[n+1] = (X == x).all(axis=1).argmax()

print(f"alp[opt] ~ alp[i[-1]]: {alp[opt]} ~ {alp[i[max(i.keys())]]}")
print(f"opt == i[-1]: {i[max(i.keys())] in (opt, 2**N - opt - 1)}")

pt.plot(alp)
for n in sorted(i.keys()):
    pt.plot(i[n], alp[i[n]], 'ro')
    pt.text(i[n], alp[i[n]], f"{n}:{alp[i[n]]:.2f}")
pt.plot([opt, opt], [alp.min(), alp.max()], 'k--')
pt.plot([2**N - opt - 1, 2**N - opt - 1], [alp.min(), alp.max()], 'k--')
pt.show()


