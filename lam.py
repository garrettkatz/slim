import numpy as np

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

    # # erase version
    # v_old = hrr_read(m, k)
    # return m - hrr_conv(k, v_old) + hrr_conv(k, v)

    # no erase version
    return m + hrr_conv(k, v)

class HRRCodec:
    @profile
    def __init__(self, symbols, dimension):
        # initializes codec with embedding vectors of given dimension for each symbol in symbols list
        self.symbols = symbols
        
        # in HRRs, keys and values need to be different vectors
        # this is because binding is symmetric, footnote 2 on p. 2 (624)
        # otherwise e.g. if M[1] = 3 and M[3] = 2, unbinding 3 will return superposition of 1 and 2
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
def run_trial(num_symbols, dimension, num_writes, initialize, write, read, Codec, replace=True):
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
        print(f"write {t} of {num_writes}...")

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

    accuracies = run_trial(
        num_symbols = 100,
        dimension = 1024, # works well when this is much larger than number of symbols
        num_writes = 50, # try (over)-writing memory this many times

        # # setup function handles for VSA method to be tested
        # initialize = lam_initialize,
        # write = lam_write,
        # read = lam_read,
        # Codec = LAMCodec,

        # setup function handles for VSA method to be tested
        initialize = hrr_initialize,
        write = hrr_write,
        read = hrr_read,
        Codec = HRRCodec,

        replace=False, # kind of "warm-up" appears when replace=True, maybe recovering from early overwrite
    )

    # show accuracy curve
    import matplotlib.pyplot as pt
    pt.plot(accuracies)
    pt.xlabel("Number of writes")
    pt.ylabel("Retrieval accuracy")
    pt.show()



