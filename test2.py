from keras.layers import (
    Input,
    Dense,
    Embedding,
    Concatenate,
    Flatten,
    Average,
    Reshape,
    Lambda,
    Activation,
)
from keras.models import Model
import keras
import keras.backend as K
import numpy as np
from nce import NCE

def build(num_items, k, num_classes):
    inputs = Input(shape=(1,), dtype="int32", name="iids")
    targets = Input(shape=(1,), dtype="int32", name="cls_ids")

    items_emb = Embedding(num_items, k, input_length=1)
    selected_item = Flatten()(items_emb(inputs))

    hidden = Dense(k, name="hidden")(selected_item)

    logits = NCE(num_classes, name="nce")([hidden, targets])

    model = keras.models.Model([inputs, targets], logits)
    model.compile(optimizer="adam", loss=None)

    return model

NUM_ITEMS = 1000000
D = 5
NUM_CLASSES = 1000000
SAMPLES = 128

x = np.random.random_integers(NUM_ITEMS, size=SAMPLES)
y = np.random.random_integers(NUM_CLASSES, size=SAMPLES)

X = [x, y]

model = build(NUM_ITEMS, D, NUM_CLASSES)
model.fit(x=X)

# model.summary()
