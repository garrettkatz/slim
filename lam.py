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
    def encode(self, symbol):
        # return the vector embedding of given symbol
        return self.embeddings[self.symbols.index(symbol)]

    @profile
    def decode(self, vector):
        # return the symbol whose embedding is most similar (in dot-product) to given vector
        return self.symbols[(self.embeddings @ vector).argmax()]

@profile
def run_trial(num_symbols, dimension, num_writes, initialize, write, read, Codec):
    # runs a sequence of VSA memory (over)writes and saves retrieval accuracy over time
    # returns the list of accuracies
    # num_symbols, dimension, num_writes are ints
    # initialize, write, read are VSA-method-specific function handles
    # Codec is a VSA-specific embedding codec class

    # setup some arbitrary symbols and codec
    symbols = [str(i) for i in range(num_symbols)]
    codec = Codec(symbols, dimension)

    # initialize linear associative matrix to all zeros
    memory = initialize(dimension)

    # initialize symbolic reference memory, also a key-value lookup
    reference_memory = {}

    # record accuracy over time as more writes are made
    accuracies = []

    # perform a series of writes
    for t in range(num_writes):
        print(f"write {t} of {num_writes}...")

        # choose a new random memory to write
        key = np.random.choice(symbols)
        val = np.random.choice(symbols)
        reference_memory[key] = val

        # encode and write the memory
        key_vec = codec.encode(key)
        val_vec = codec.encode(val)
        memory = write(memory, key_vec, val_vec)

        # check how many memories are accurately stored
        num_correct = 0
        for key, val in reference_memory.items():
            # this inner loop is the main bottleneck, could sub-sample key-val pairs to speed up

            # encode current key and read it from memory
            key_vec = codec.encode(key)
            result_vec = read(memory, key_vec)

            # decode result and check against symbolic memory
            result = codec.decode(result_vec)
            if result == val: num_correct += 1

        # update retrieval accuracy
        accuracies.append(num_correct / len(reference_memory))

    return accuracies

if __name__ == "__main__":

    accuracies = run_trial(
        num_symbols = 50,
        dimension = 50, # works well when this is much larger than number of symbols
        num_writes = 512, # try (over)-writing memory this many times

        # setup function handles for VSA method to be tested
        initialize = lam_initialize,
        write = lam_write,
        read = lam_read,
        Codec = LAMCodec,
    )

    # show accuracy curve
    import matplotlib.pyplot as pt
    pt.plot(accuracies)
    pt.xlabel("Number of writes")
    pt.ylabel("Retrieval accuracy")
    pt.show()



