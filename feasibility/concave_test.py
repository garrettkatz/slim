import numpy as np

N = 2

# all vectors in positive orthant will have positive dot
def sample():
    x = np.fabs(np.random.randn(10))
    return x

def cos(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def diff(x, y):
    # return (np.linalg.norm(x) * np.linalg.norm(y)) - np.dot(x, y)
    return (np.linalg.norm(x) * np.linalg.norm(y))**2 - np.dot(x, y)**2

numreps = 100

for rep in range(numreps):

    u, v, u_, v_ = (sample() for _ in range(4))
    
    a = np.random.rand()

    au = a*u + (1-a)*u_
    av = a*v + (1-a)*v_

    # fa = cos(au, av)
    # af = a*cos(u, v) + (1-a)*cos(u_, v_)
    # concave = (fa >= af)

    # if not concave:
    #     print(rep)
    #     print(u)
    #     print(v)
    #     print(u_)
    #     print(v_)
    #     print(cos(u, v), cos(u_, v_))
    #     print(fa, af)

    #     import matplotlib.pyplot as pt
    #     pt.plot([0, u[0]], [0, u[1]], 'b-')
    #     pt.plot([0, v[0]], [0, v[1]], 'r-')
    #     pt.plot([0, u_[0]], [0, u_[1]], 'b:')
    #     pt.plot([0, v_[0]], [0, v_[1]], 'r:')
    #     pt.plot([0, au[0]], [0, au[1]], 'b-.')
    #     pt.plot([0, av[0]], [0, av[1]], 'r-.')
    #     pt.show()
    
    # assert concave

    fa = diff(au, av)
    af = a*diff(u, v) + (1-a)*diff(u_, v_)
    convex = (fa <= af)

    if not convex:
        print(rep)
        print(u)
        print(v)
        print(u_)
        print(v_)
        print(diff(u, v), diff(u_, v_))
        print(fa, af)

        import matplotlib.pyplot as pt
        pt.plot([0, u[0]], [0, u[1]], 'b-')
        pt.plot([0, v[0]], [0, v[1]], 'r-')
        pt.plot([0, u_[0]], [0, u_[1]], 'b:')
        pt.plot([0, v_[0]], [0, v_[1]], 'r:')
        pt.plot([0, au[0]], [0, au[1]], 'b-.')
        pt.plot([0, av[0]], [0, av[1]], 'r-.')
        pt.show()
    
    assert convex

