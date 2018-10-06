from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model

import keras.backend as K

import numpy as np

from nce import NCE


def build(NUM_ITEMS, num_users, k):

    iid = Input(shape=(1,), dtype="int32", name="iids")
    targets = Input(shape=(1,), dtype="int32", name="target_ids")

    item_embedding = Embedding(
        input_dim=NUM_ITEMS, output_dim=k, input_length=1, name="item_embedding"
    )
    selected_items = Flatten()(item_embedding(iid))

    h1 = Dense(k // 2, activation="relu", name="hidden")(selected_items)
    sig = Dense(NUM_ITEMS, activation="softmax", name="softmax")(h1)

    sm_logits = NCE(num_users, name="nce")([h1, targets])

    model = Model(inputs=[iid, targets], outputs=[sm_logits])
    return model


K = 10
SAMPLE_SIZE = 10000
num_items = 10000
NUM_USERS = 1000000

model = build(num_items, NUM_USERS, K)
model.compile(optimizer="adam", loss=None)
model.summary()

x = np.random.random_integers(num_items - 1, size=SAMPLE_SIZE)
y = np.ones(SAMPLE_SIZE)
X = [x, y]
print(x.shape, y.shape)

model.fit(x=X, batch_size=100, epochs=1)


TEST_SAMPLES = 10
x = np.random.random_integers(num_items - 1, size=TEST_SAMPLES)
y = np.ones(TEST_SAMPLES)
X = [x, y]

preds = model.predict(X)
print(preds.shape)

idx = np.argmax(preds, axis=1)
# print(idx, preds[idx])
