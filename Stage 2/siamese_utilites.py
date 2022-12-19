# from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np


def get_batch(triplets_list, batch_size, preprocess=False):
    batches_count = len(triplets_list) // batch_size
    triplet_index = 0
    bateches_counter = 0
    while bateches_counter < batches_count:  # no of batches
        j = bateches_counter * batch_size  # index of first triplet in the batch
        while j < (bateches_counter + 1) * batch_size and j < len(triplets_list):
            anchor = []
            positive = []
            negative = []
            j += 1
            anch, posit, negat = triplets_list[triplet_index]
            triplet_index += 1
            anch = np.array(anch, dtype=object)
            posit = np.array(posit, dtype=object)
            negat = np.array(negat, dtype=object)
            # if preprocess:
            #     anch = preprocess_input(anchor)
            #     posit = preprocess_input(positive)
            #     negat = preprocess_input(negative)
            anchor.append(anch)
            positive.append(posit)
            negative.append(negat)
        yield ([anchor, positive, negative])
        # batches[bateches_counter][increment]=triplets_list[triplet_index]
        bateches_counter += 1
