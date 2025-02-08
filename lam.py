import pickle as pk
import numpy as np
from hadamard_cleanup import sylvester, cleanup

# for profiling the script, or not
try: profile
except NameError: profile = lambda x: x

@profile
def lam_initialize(dimension):
    # returns initial memory state for given dimension
    return np.zeros((dimension, dimension))

@profile
def lam_read(m, k):
    # returns value vector currently bound to key vector k in memory m
    return np.sign(m @ k)

@profile
def lam_write(m, k, v):
    # binds value vector v to key vector k and stores result in memory m
    # returns updated memory

    # this rule is NOT commutative, getting the order of k,v wrong breaks it
    return m  - np.outer(np.sign(m @ k), k) + np.outer(k, v)

class LAMCodec:
    @profile
    def __init__(self, symbols, dimension):
        # initializes codec with embedding vectors of given dimension for each symbol in symbols list
        self.symbols = symbols
        self.embeddings = np.random.choice((-1, +1), size=(len(symbols), dimension))

    @profile
    def encode_key(self, symbol):
        # return the vector embedding of given symbol as a key
        return self.embeddings[self.symbols.index(symbol)]

    @profile
    def decode_key(self, vector):
        # return the symbol whose embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.embeddings @ vector).argmax()]

    # for LAM same vectors can be used as keys and values
    @profile
    def encode_val(self, symbol): return self.encode_key(symbol)
    @profile
    def decode_val(self, vector): return self.decode_key(vector)

@profile
def hrr_initialize(dimension):
    # returns initial memory state for given dimension
    return np.zeros(dimension)

@profile
def hrr_conv(c, x):

    # # slow direct method
    # idx = np.arange(len(x))
    # idx = idx[:,np.newaxis] - idx
    # t = x[idx] @ c

    # fast fft method
    # from https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
    t2  = np.real(np.fft.ifft( np.fft.fft(c)*np.fft.fft(x) ))
    # assert np.allclose(t, t2) # made sure they match

    return t2

@profile
def hrr_read(m, k):
    # returns value vector currently bound to key vector k in memory m
    k_inv = np.roll(k[::-1], 1)
    return hrr_conv(k_inv, m)

@profile
def hrr_write(m, k, v):
    # binds value vector v to key vector k and stores result in memory m
    # returns updated memory
    # does not try to erase earlier writes with the same key
    return m + hrr_conv(k, v)

@profile
def hrr_overwrite(m, k, v):
    # binds value vector v to key vector k and stores result in memory m
    # returns updated memory
    # tries to erase earlier writes with the same key
    v_old = hrr_read(m, k)
    return m - hrr_conv(k, v_old) + hrr_conv(k, v)

class HRRCodec:
    @profile
    def __init__(self, symbols, dimension):
        # initializes codec with embedding vectors of given dimension for each symbol in symbols list
        self.symbols = symbols
        
        # in HRRs, keys and values need to be different vectors
        # this is because binding is symmetric, footnote 2 on p. 2 (624)
        # otherwise e.g. if M[1] = 3 and M[3] = 2, unbinding 3 will return superposition of 1 and 2
        self.key_embeddings = np.random.randn(len(symbols), dimension) / dimension ** 0.5
        self.val_embeddings = np.random.randn(len(symbols), dimension) / dimension ** 0.5

        # this was a bug, but oddly worked better with overwrite=True
        self.key_embeddings = np.random.randn(len(symbols), dimension) / dimension
        self.val_embeddings = np.random.randn(len(symbols), dimension) / dimension

    @profile
    def encode_key(self, symbol):
        # return the vector embedding of given symbol as a key
        return self.key_embeddings[self.symbols.index(symbol)]
    @profile
    def encode_val(self, symbol):
        # return the vector embedding of given symbol as a value
        return self.val_embeddings[self.symbols.index(symbol)]

    @profile
    def decode_key(self, vector):
        # return the symbol whose key embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]
    @profile
    def decode_val(self, vector):
        # return the symbol whose value embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]

@profile
def binary_hrr_read(m, k):
    # returns value vector currently bound to key vector k in memory m
    k_inv = np.roll(k[::-1], 1)
    v_noisy = hrr_conv(k_inv, m)
    v_clean = np.sign(v_noisy) / v_noisy.size ** 0.5
    return v_clean

@profile
def binary_hrr_overwrite(m, k, v):
    # binds value vector v to key vector k and stores result in memory m
    # returns updated memory
    # tries to erase earlier writes with the same key
    v_old = binary_hrr_read(m, k)
    return m - hrr_conv(k, v_old) + hrr_conv(k, v)

class BinaryHRRCodec:
    @profile
    def __init__(self, symbols, dimension):
        # initializes codec with embedding vectors of given dimension for each symbol in symbols list
        self.symbols = symbols
        
        # in HRRs, keys and values need to be different vectors
        # this is because binding is symmetric, footnote 2 on p. 2 (624)
        # otherwise e.g. if M[1] = 3 and M[3] = 2, unbinding 3 will return superposition of 1 and 2
        self.key_embeddings = np.random.choice([-1,+1], size=(len(symbols), dimension)) / dimension ** 0.5
        self.val_embeddings = np.random.choice([-1,+1], size=(len(symbols), dimension)) / dimension ** 0.5

    @profile
    def encode_key(self, symbol):
        # return the vector embedding of given symbol as a key
        return self.key_embeddings[self.symbols.index(symbol)]
    @profile
    def encode_val(self, symbol):
        # return the vector embedding of given symbol as a value
        return self.val_embeddings[self.symbols.index(symbol)]

    @profile
    def decode_key(self, vector):
        # return the symbol whose key embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]
    @profile
    def decode_val(self, vector):
        # return the symbol whose value embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]

@profile
def hadamard_hrr_read(m, k):
    # returns value vector currently bound to key vector k in memory m
    k_inv = np.roll(k[::-1], 1)
    v_noisy = hrr_conv(k_inv, m)
    v_clean = cleanup(v_noisy)
    return v_clean

@profile
def hadamard_hrr_overwrite(m, k, v):
    # binds value vector v to key vector k and stores result in memory m
    # returns updated memory
    # tries to erase earlier writes with the same key
    v_old = hadamard_hrr_read(m, k)
    return m - hrr_conv(k, v_old) + hrr_conv(k, v)

class HadamardHRRCodec:
    @profile
    def __init__(self, symbols, dimension):
        # initializes codec with embedding vectors of given dimension for each symbol in symbols list
        self.symbols = symbols
        
        # keys must have HRR stats, but values can be hadamard
        self.key_embeddings = np.random.randn(len(symbols), dimension) / dimension ** 0.5
        self.val_embeddings = sylvester(int(np.ceil(np.log2(dimension))))[:len(symbols)]

    @profile
    def encode_key(self, symbol):
        # return the vector embedding of given symbol as a key
        return self.key_embeddings[self.symbols.index(symbol)]
    @profile
    def encode_val(self, symbol):
        # return the vector embedding of given symbol as a value
        return self.val_embeddings[self.symbols.index(symbol)]

    @profile
    def decode_key(self, vector):
        # return the symbol whose key embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]
    @profile
    def decode_val(self, vector):
        # return the symbol whose value embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.val_embeddings @ vector).argmax()]


@profile
def run_trial(num_symbols, dimension, num_writes, initialize, write, read, Codec, replace=True, verbose=True):
    # runs a sequence of VSA memory (over)writes and saves retrieval accuracy over time
    # returns the list of accuracies
    # num_symbols, dimension, num_writes are ints
    # initialize, write, read are VSA-method-specific function handles
    # Codec is a VSA-specific embedding codec class
    # replace is whether keys are sampled with replacement (necessitating overwrites)

    # setup some arbitrary symbols and codec
    symbols = [str(i) for i in range(num_symbols)]
    codec = Codec(symbols, dimension)

    # initialize linear associative matrix to all zeros
    memory = initialize(dimension)

    # initialize symbolic reference memory, also a key-value lookup
    reference_memory = {}

    # record accuracy over time as more writes are made
    accuracies = []

    # pre-permute keys for sampling without replacement condition (no overwrites)
    if not replace: keys = np.random.permutation(symbols)

    # perform a series of writes
    for t in range(num_writes):
        if verbose: print(f"write {t} of {num_writes}...")

        # choose a new random memory to write
        if replace:
            key = np.random.choice(symbols) # with replacement
        else:
            key = keys[t % len(keys)] # without replacement, at least before t == len(symbols)
        val = np.random.choice(symbols)
        reference_memory[key] = val

        # encode and write the memory
        key_vec = codec.encode_key(key)
        val_vec = codec.encode_val(val)
        memory = write(memory, key_vec, val_vec)

        # check how many memories are accurately stored
        num_correct = 0
        for key, val in reference_memory.items():
            # this inner loop is the main bottleneck, could sub-sample key-val pairs to speed up

            # encode current key and read it from memory
            key_vec = codec.encode_key(key)
            result_vec = read(memory, key_vec)

            # decode result and check against symbolic memory
            result = codec.decode_val(result_vec)
            if result == val: num_correct += 1

        # update retrieval accuracy
        accuracies.append(num_correct / len(reference_memory))

    return accuracies

if __name__ == "__main__":

    num_reps = 10 # this many random repetitions of the experiment
    num_symbols = 32 # this many distinct symbols (addresses and values)
    dimensions = (256, 512, 1024, 2048) # try these vector dimensions
    num_writes = 50 # try (over)-writing memory this many times
    replace = True # whether to sample addresses with replacement (whether to overwrite early in the trial)

    do_run = True
    do_show = True

    if do_run:
        accuracies = {}
        for overwrite in (False, True):
            for dimension in dimensions:
                accuracies[overwrite, dimension] = []
    
                for rep in range(num_reps):
                    print(f"{overwrite=}, {dimension=}, {rep=}")
    
                    accuracy_curve = run_trial(
                        num_symbols,
                        dimension, # works well when this is much larger than number of symbols
                        num_writes,
                
                        # # setup function handles for VSA method to be tested
                        # lam_initialize,
                        # lam_write,
                        # lam_read,
                        # LAMCodec,
                
                        hrr_initialize,
                        hrr_overwrite if overwrite else hrr_write,
                        hrr_read,
                        HRRCodec,

                        # hrr_initialize,
                        # binary_hrr_overwrite if overwrite else hrr_write,
                        # binary_hrr_read,
                        # BinaryHRRCodec,

                        # hrr_initialize,
                        # hadamard_hrr_overwrite if overwrite else hrr_write,
                        # hadamard_hrr_read,
                        # HadamardHRRCodec,
                
                        replace, # mini "warm-up" can appear when replace=True, maybe recovering from early overwrite
                        verbose=False,
                    )
    
                    accuracies[overwrite, dimension].append(accuracy_curve)

        with open(f"lam_{num_symbols}_symbols_{num_writes}_writes_replace_{replace}.pkl", "wb") as f: pk.dump(accuracies, f)

    if do_show:

        with open(f"lam_{num_symbols}_symbols_{num_writes}_writes_replace_{replace}.pkl", "rb") as f: accuracies = pk.load(f)

        import matplotlib.pyplot as pt

        # plot accuracy curves
        fig, ax = pt.subplots(2, len(dimensions), figsize=(10,4), constrained_layout=True)
        for o, overwrite in enumerate((False, True)):
            for d, dimension in enumerate(dimensions):
    
                data = np.array(accuracies[overwrite, dimension])
                ax[o,d].plot(data.T, color=(0.85,)*3)
                ax[o,d].plot(data.mean(axis=0), color='k')
                ax[o,d].set_ylim([0,1.1])

                if o == 0:
                    ax[o,d].set_title(f"{dimension=}")
                if d == 0:
                    ax[o,d].set_ylabel(f"{overwrite=}")
    
        fig.supxlabel("Number of writes")
        fig.supylabel("Retrieval accuracy")
        pt.savefig(f"lam_acc_{num_symbols}_symbols_{num_writes}_writes_replace_{replace}.pdf")
        pt.show()

        # plot number of writes until first incorrect read
        fig, ax = pt.subplots(2, len(dimensions), figsize=(10,4), constrained_layout=True)
        for o, overwrite in enumerate((False, True)):
            for d, dimension in enumerate(dimensions):
    
                data = np.array(accuracies[overwrite, dimension])
                num_correct = (data < 1).argmax(axis=1) - 1

                print(f"{overwrite=}, {dimension=}: streak = {num_correct.mean():.3f} +/- {num_correct.std():.3f}")

                ax[o,d].hist(num_correct, bins = np.arange(0, num_writes+1, 2), facecolor=(0.85,)*3, edgecolor='k')
                ax[o,d].set_xlim([0, num_writes+1])

                if o == 0:
                    ax[o,d].set_title(f"{dimension=}")
                if d == 0:
                    ax[o,d].set_ylabel(f"{overwrite=}")
    
        fig.supxlabel("Number of correct writes")
        fig.supylabel("Frequency")
        pt.savefig(f"lam_cor_{num_symbols}_symbols_{num_writes}_writes_replace_{replace}.pdf")
        pt.show()
        

