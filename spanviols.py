import numpy as np
import matplotlib.pyplot as pt

# N, inspan, outspan
# numbers copied from running python boundaries.py N
data = np.array([
    [3, 7, 0],
    [4, 18, 0],
    [5, 61, 2],
    [6, 216, 28],
    [7, 1444, 282],
])

pt.figure(figsize=(3,3))
pt.plot(data[:,0], data[:,1], 'o-', label="In span")
pt.plot(data[2:,0], data[2:,2], 'o-', label="Out span") # exclude log(0)
pt.xlabel("Number of neurons")
pt.ylabel("Threshold map pairs")
pt.yscale("log")
pt.xticks(data[:,0])
pt.legend()
pt.tight_layout()
pt.show()
