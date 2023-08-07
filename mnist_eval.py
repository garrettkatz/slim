import os
import numpy as np
import matplotlib.pyplot as pt
from sklearn.svm import LinearSVC
import torch as tr
import torchvision as tv
from torchvision.transforms import ToTensor
from softfit import SpanRule, form_str

if __name__ == "__main__":

    # define function for perceptron learning rule
    def perceptron_rule(w, x, y, N):
        return w + (y - np.sign(w @ x)) * x

    train = tv.datasets.MNIST(
        root = os.path.join(os.environ["HOME"], "Downloads"),
        train = True,
        transform = tv.transforms.ToTensor(),
        download = True)

    test = tv.datasets.MNIST(
        root = os.path.join(os.environ["HOME"], "Downloads"),
        train = False,
        transform = tv.transforms.ToTensor(),
        download = True)

    N = 28**2 # MNIST is 1x28x28 images
    classes = (1, 2)
    max_examples = 100#15000

    num_examples = 0
    for i in range(len(train)):
        x, y = train[i]
        if y in classes: num_examples += 1
        if num_examples == max_examples: break
    print(num_examples)

    X = np.empty((num_examples, N), dtype=int)
    Y = np.empty(num_examples, dtype=int)
    n = 0
    for i in range(len(train)):
        x, y = train[i]
        if y not in classes: continue

        x = x.flatten().numpy()
        X[n] = (x > (x.max() + x.min())/2).astype(int)
        Y[n] = (-1)**int(y == classes[0])

        n += 1
        if n == num_examples: break

    print(X.shape, Y.shape)

    # random shuffle
    idx = np.random.permutation(X.shape[0])
    X, Y = X[idx], Y[idx]

    # ##### SVC

    svc = LinearSVC(dual='auto', fit_intercept=False)
    svc.fit(X, Y)
    pred = svc.predict(X)
    svm_acc = (pred == Y).mean()
    print(f"svm train acc = {svm_acc}")

    # ##### perceptron

    w = np.zeros(N)
    for x, y in zip(X, Y):
        w = perceptron_rule(w, x, y, N)
    pred = np.sign(X @ w)
    plr_acc = (pred == Y).mean()
    print(f"plr train acc = {plr_acc}")

    # pt.imshow(w.reshape(28,28))
    # pt.show()

    # ##### soft

    model = tr.load('softfit.pt')
    print("form", form_str(model.alpha.harden()), form_str(model.beta.harden()))

    def soft_rule(w, x, y, N):
        inputs = {
            'w': tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)),
            'x': tr.tensor(x, dtype=tr.float32).view(1,N),
            'y': tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # broadcast
            '1': tr.ones(1,N), # broadcast
        }
        with tr.no_grad():
            w_new = model(inputs)
        return w_new.clone().squeeze().numpy()

    w = np.zeros(N)
    for i, (x, y) in enumerate(zip(X, Y)):
        w = soft_rule(w, x, y, N)
        # if i % 10 == 0: print(f"soft {i}", np.fabs(w).max())
        print(f"soft {i}", np.fabs(w).max())
    pred = np.sign(X @ w)
    sft_acc = (pred == Y).mean()
    print(f"sft train acc = {sft_acc}")
