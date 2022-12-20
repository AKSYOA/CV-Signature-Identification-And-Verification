# from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import tensorflow as tf
from random import shuffle

from tensorflow.keras import backend, layers, metrics
from keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam

from opt_einsum.backends import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential


def get_triplet(data):
    triplets = []
    classes = ['A', 'B', 'C', 'D', 'E']
    for i in range(len(classes)):
        Class = classes[i]
        for image in data:
            if image[2].__contains__(Class):
                anchor = image[0]
                label_anchor = image[1]
                name = image[2]

        for pos in data:
            if pos[2].__contains__(Class) and label_anchor == pos[1] and name != pos[2]:
                positive = pos
                for neg in data:
                    if neg[2].__contains__(Class) and label_anchor != neg[1]:
                        negative = neg
                        # print("anchor", label_anchor, " ", name)
                        # print("postive", pos[1], " ", pos[2])
                        # print("negative", neg[1], " ", neg[2], "\n")
                        triplets.append([anchor, positive, negative])
                        shuffle(triplets)
    return triplets


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
        yield [anchor, positive, negative]
        # batches[bateches_counter][increment]=triplets_list[triplet_index]
        bateches_counter += 1


def cnn_model(image_size):
    pretrained_model = Xception(
        input_shape=image_size,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    for i in range(len(pretrained_model) - 27):
        pretrained_model.layers[i].trainable = False
        cnn = Sequential([
            pretrained_model,
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(512, activation='relu'),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.Dense(256, activation="relu"),
            tensorflow.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], name="Encode_Model")

    return cnn


class DistanceLayer(layers.Layer):  #######leh 3mltha fe class w eh momizat ani a4t8al b class
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


def get_siamese_network(image_size=(128, 128)):
    model = cnn_model(image_size)  # make model
    anchor = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='anchor')
    positive = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='positive')
    negative = tensorflow.keras.layers.Input(input_size=(image_size, image_size), name='negative')
    output_anchor = model(anchor)  # inputLayers to model
    output_positive = model(positive)
    output_negative = model(negative)

    distances = DistanceLayer()(output_anchor, output_positive, output_negative)
    siamese_network = Model(
        inputs=[anchor, positive, negative],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network


class siamese_model(Model):
    def __init__(self, siamese_network, margin=1.0):
        self.margin = margin
        self.siamese_network = siamese_network
        self.error_improver = metrics.Mean(name="loss")

    def call(self, input):
        return self.siamese_network(input)

    def get_loss(self, triplets):
        anchor_positive, anchor_negative = self.siamese_network(triplets)
        return tf.reduce_sum(max(anchor_positive - anchor_negative + self.margin, 0))

    def train(self,train_triplets):
        with tf.GradientTape as tape:
            loss = self.get_loss(train_triplets)
        gradient = tape.gradient(loss,self.siamese_network.trainable_weights) # dloss/dW
        self.optimizer.apply_gradients(zip(gradient,self.siamese_network.trainable_weights))
        self.error_improver.update_state(loss)
        return {"loss": self.error_improver.result()}

    def test(self,test_triplets):
        loss= self.get_loss(test_triplets)
        self.error_improver.update_state(loss)

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]
