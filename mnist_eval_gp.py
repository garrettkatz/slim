import os
import pickle as pk
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as pt
from sklearn.svm import LinearSVC
import torch as tr
import torchvision as tv
from torchvision.transforms import ToTensor
import softform_dense as sfd
import hardform_dense as hfd

# assumes at least num examples with specified classes in dataset
def get_class_examples(dataset, classes, num):
    count = 0
    for i in np.random.permutation(len(dataset)):
        x, y = dataset[i]
        if y not in classes: continue
        yield x, y
        count += 1
        if count == num: break

if __name__ == "__main__":

    do_eval = True
    N = 1 + 28**2 # MNIST is 1x28x28 images, plus bias
    num_examples = {"train": 100, "test": 100}
    num_reps = 30

    # define function for perceptron learning rule
    def perceptron_rule(w, x, y, N):
        return w + (y - np.sign(w @ x)) * x

    datasets = {}
    for train in (False, True):
        key = ["test", "train"][train]
        datasets[key] = tv.datasets.MNIST(
            root = os.path.join(os.environ["HOME"], "Downloads"),
            train = train,
            transform = tv.transforms.ToTensor(),
            download = True)

    if do_eval:

        accs = {(key,mod): np.empty(num_reps) for key in ("train","test") for mod in ("svm", "perceptron", "gp")}
        for rep in range(num_reps):
    
            classes = tuple(np.random.choice(10, size=2, replace=False))
            print(f"{rep} of {num_reps}: classes:", classes)
        
            X, Y = {}, {}
            for key in ["train", "test"]:
                X[key] = np.empty((num_examples[key], N), dtype=int)
                Y[key] = np.empty(num_examples[key], dtype=int)
                for n, (x, y) in enumerate(get_class_examples(datasets[key], classes, num_examples[key])):
                    x = x.flatten().numpy()
                    # X[key][n] = (x > (x.max() + x.min())/2).astype(int)
                    X[key][n, 0] = -1 # bias to match training data
                    X[key][n, 1:] = (-1)**(x < (x.max() + x.min())/2).astype(int)
                    Y[key][n] = (-1)**int(y == classes[0])
        
            # ##### SVC
        
            svc = LinearSVC(dual='auto', fit_intercept=False)
            svc.fit(X['train'], Y['train'])
            for key in ["train", "test"]:
                pred = svc.predict(X[key])
                svm_acc = (pred == Y[key]).mean()
                print(f"svm {key} acc = {svm_acc}")
                accs[(key,"svm")][rep] = svm_acc
        
            # ##### perceptron
        
            w = np.zeros(N)
            for x, y in zip(X['train'], Y['train']):
                w = perceptron_rule(w, x, y, N)
            for key in ["train", "test"]:
                pred = np.sign(X[key] @ w)
                perceptron_acc = (pred == Y[key]).mean()
                print(f"perceptron {key} acc = {perceptron_acc}")
                accs[(key,"perceptron")][rep] = perceptron_acc
        
            # pt.imshow(w.reshape(28,28))
            # pt.show()
        
            # ##### gp

            with open("genprog.pkl", "rb") as f:
                form, _ = pk.load(f)
            print("form", sfd.form_str(form))
        
            def gp_rule(w, x, y, N):
                inputs = tr.stack([
                    tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)), # w
                    tr.tensor(x, dtype=tr.float32).view(1,N), # x
                    tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # y
                    tr.ones(1,N), # 1
                ])
                w_new = hfd.tree_eval(form, inputs)
                w_new = tr.nn.functional.normalize(w_new)
                return w_new.clone().squeeze().numpy()
        
            w = np.zeros(N)
            for i, (x, y) in enumerate(zip(X['train'], Y['train'])):
                w = gp_rule(w, x, y, N)
                # if i % 10 == 0: print(f"gp {i}", np.fabs(w).max())
                # print(f"gp {i}", np.fabs(w).max())
        
            for key in ["train", "test"]:
                pred = np.sign(X[key] @ w)
                gp_acc = (pred == Y[key]).mean()
                print(f"gp {key} acc = {gp_acc}")
                accs[(key,"gp")][rep] = gp_acc

        with open("mnist_eval_gp.pkl","wb") as f:
            pk.dump(accs, f)

    with open("mnist_eval_gp.pkl","rb") as f:
        accs = pk.load(f)

    result = ttest_1samp(accs['test','gp'], 0.5, alternative='greater')
    print(f"stat={result.statistic}, pval={result.pvalue}")

    # pt.plot([0]*num_reps, accs[('train','gp')], 'r.')
    # pt.plot([1]*num_reps, accs[('test','gp')], 'b.')
    # pt.scatter(0, np.mean(accs[('train','gp')]), 50, color='r')
    # pt.scatter(1, np.mean(accs[('test','gp')]), 50, color='b')
    # pt.scatter(0, np.mean(accs[('train','gp')]) - np.std(accs[('train','gp')]), 50, color='r')
    # pt.scatter(1, np.mean(accs[('test','gp')]) - np.std(accs[('test','gp')]), 50, color='b')
    # pt.plot([0,1],[.5,.5], 'k:')
    # pt.xticks([0,1], ['train','test'], rotation=90)
    # pt.show()

    pt.figure(figsize=(4,3))
    # pt.hist(accs[('train','gp')], bins = np.linspace(0, 1.0, 10), align='left', rwidth=0.5, label="Train")
    # pt.hist(accs[('test','gp')], bins = np.linspace(0, 1.0, 10), align='mid', rwidth=0.5, label="Test")
    # # pt.xlim([.3, 1.0])
    pt.hist(accs[('train','gp')], align='left', rwidth=0.5, label="Train")
    pt.hist(accs[('test','gp')], align='mid', rwidth=0.5, label="Test")
    pt.xlabel("Accuracy")
    pt.ylabel("Frequency")
    pt.legend()
    pt.tight_layout()
    pt.savefig("mnist_gp.pdf")
    pt.show()

