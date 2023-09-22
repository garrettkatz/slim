import numpy as np
import tensorflow as tf

np.set_printoptions(precision=4)
import scipy as sp
from scipy.stats import ttest_1samp

import matplotlib.pyplot as pt

from sklearn.utils import shuffle


def gen_mnist_classif_data(digit=2, size = 100):
    # get digit-specific MNIST data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # x_train, y_train = x_train[y_train==digit], y_train[y_train==digit]
    #y_train = (y_train == digit).astype("int8")  # np.ones_like(y_train, dtype='int8')
    # y_train = y_train.reshape((-1,1))
    # y_train = tf.one_hot(y_train, depth=2)
    #y_test = (y_test == digit).astype("int8")
    # y_test = y_test.reshape((-1, 1))
    # y_test = tf.one_hot(y_test, depth=2)
    # y_train =  tf.one_hot(y_train, depth=2)
    # y_test = tf.one_hot(y_test, depth=2)
    y_train = (y_train == digit).astype("float32")
    y_test = (y_test == digit).astype("float32")
    x_train, y_train = shuffle(x_train, y_train)
    return (x_train[:size], y_train[:size]), (x_test, y_test)


def get_model():
    # a 'linear' model in TensorFlow

    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])

    optimizer = tf.keras.optimizers.Adam(0.001)  # not needed, as the optimization is carried out manually
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=[tf.keras.metrics.binary_accuracy])

    print(model.summary())
    return model, loss_fn


def get_stoch_gradient(model, loss_fn, inputs, labels):
    # stochastic gradient with respect to a single (input, output pair), note usage in the code below

    #print(model(inputs), model(inputs).shape, labels.shape)
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        pred_loss = loss_fn(labels, predictions)
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    return np.hstack([arr.numpy().ravel() for arr in gradients])


def get_gradient(model, inputs):
    # gradient with respect to a model output, refered to as \del f_i in the paper

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
    gradients = tape.gradient(predictions, model.trainable_variables)
    return np.hstack([arr.numpy().ravel() for arr in gradients])


def get_projection(basis, vec):
    # project a vector onto a given basis

    if not (len(basis) > 0):
        return np.zeros_like(vec)

    vec_sum = np.zeros_like(vec)
    for col in range(basis.shape[1]):
        col = basis[:, col].ravel()
        vec_projected = np.dot(vec, col) * col / (np.linalg.norm(col, ord=2) ** 2 + 1e-10)
        vec_sum += vec_projected

    return vec_sum


def set_model_weights(vec, model):
    # set model weights

    start = 0
    end = None
    for tv in model.trainable_variables:
        size = tf.math.reduce_prod(tv.shape)
        end = start + size
        tv.assign(tf.reshape(vec[start:end].astype('float32'), shape=tv.shape))
        start = end


def eval_model(model, x_train, y_train, x_test, y_test, return_arr=False, verbose=False):
    # eval_train, eval_test = (model.evaluate(x_train, y_train, verbose=verbose),
    #                          model.evaluate(x_test, y_test, verbose=verbose))
    eval_train = (tf.keras.losses.BinaryCrossentropy()(y_train, model.predict(x_train, verbose=False)),
                  tf.keras.metrics.BinaryAccuracy()(y_train, model.predict(x_train, verbose=False)))
    eval_test = (tf.keras.losses.BinaryCrossentropy()(y_test, model.predict(x_test, verbose=False)),
                 tf.keras.metrics.BinaryAccuracy()(y_test, model.predict(x_test, verbose=False)))
    if verbose:
        print("Eval (loss, acc)", "\t", f"Train: {eval_train}", f"Test: {eval_test}")
    if return_arr:
        return eval_train, eval_test


def ORFit(ds_size=100, m=10, digit=2):
    n_ds = ds_size  # number of elements in the data sequence
    model, loss_func = get_model()
    (x_train, y_train), (x_test, y_test) = gen_mnist_classif_data(digit=digit, size = ds_size)

    logs = []  # [eval_model(model, x_train, y_train, x_test, y_test, return_arr=True)]

    U = np.array([])
    E = np.array([])
    w = np.hstack([arr.numpy().ravel() for arr in model.trainable_variables])

    for i in range(n_ds):

        g = get_stoch_gradient(model, loss_func, np.expand_dims(x_train[i], axis=0),
                               np.reshape(y_train[i], (1, -1)))
        g_prime = g - get_projection(U, g)
        v_prime = (get_gradient(model, np.expand_dims(x_train[i], axis=0)) \
                   - get_projection(U, get_gradient(model, np.expand_dims(x_train[i], axis=0))))

        if i <= (m - 1):
            if i == 0:
                U = np.expand_dims(v_prime, axis=1)
            else:
                U = np.append(U, np.expand_dims(v_prime, axis=1), axis=1)

            if i == m - 1:
                temp_shape = U.shape
                U, E_vals, V_temp = sp.linalg.svd(U)
                E = sp.linalg.diagsvd(E_vals, *temp_shape)


        else:
            u = v_prime / (1e-10 + np.linalg.norm(v_prime, ord=2))

            idx_resample = E
            idx_resample = np.append(idx_resample, np.zeros(shape=(idx_resample.shape[0], 1)), axis=1)
            idx_resample = np.append(idx_resample, np.zeros(shape=(1, idx_resample.shape[1])), axis=0)
            idx_resample[-1, -1] = np.dot(u, v_prime)

            U_prime, E_vals, V_temp = sp.linalg.svd(idx_resample)
            E = sp.linalg.diagsvd(E_vals, *idx_resample.shape)

            U = np.append(U, np.expand_dims(u, axis=1), axis=1)
            U = np.matmul(U, U_prime)
            U, E = U[:, :m], E[:m, :]

        eta = (tf.squeeze(model(np.expand_dims(x_train[i], 0)) - y_train[i]).numpy()) / (
                    1e-10 + np.dot(get_gradient(model, np.expand_dims(x_train[i], 0)),
                                   g_prime))

        w = w - eta * g_prime

        set_model_weights(w, model)

        if i > 0 and i % 10 == 0:
            print(f"Iteration # {i}")
            #print()
            eval_model(model, x_train, y_train, x_test, y_test, verbose=True)
        logs.append(eval_model(model, x_train, y_train, x_test, y_test, return_arr=True, verbose=False))

    return logs


logs_all = {"train": [], "test": []}
num_runs = 3
for run in range(num_runs):
    logs = np.array(ORFit(100, m=10, digit=np.random.randint(0, 10, size=1)))
    mean_tr_acc = logs[:, 0, 1].mean()
    mean_te_acc = logs[:, 1, 1].mean()
    logs_all["train"].append(mean_tr_acc)
    logs_all["test"].append(mean_te_acc)
result = ttest_1samp(logs_all['test'], 0.5, alternative='greater')
print(f"stat={result.statistic}, pval={result.pvalue}")
pt.figure(figsize=(6, 3))
pt.hist(logs_all['train'], bins=np.linspace(0, 1.0, 10), align='left', rwidth=0.5, label="Train")
pt.hist(logs_all['test'], bins=np.linspace(0, 1.0, 10), align='mid', rwidth=0.5, label="Test")
# pt.xlim([.3, 1.0])
pt.xlabel("Accuracy")
pt.ylabel("Frequency")
pt.legend()
pt.tight_layout()
pt.figtext(0.1, 0.5, f'stat={np.round(result.statistic, 4)},'
                     f' pval={np.round(result.pvalue, 6)}')
pt.savefig("mnist.pdf")
pt.show()