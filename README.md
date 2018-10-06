# Keras NCE-loss

Keras implemenation of the candidate sampling technique called Noise Contrastive Estimation (NCE). This is a Keras Layer which uses the TF implementation of NCE loss.

*Mnih, Teh. A fast and simple algorithm for training neural probabilistic language models. ICML 2012*



NCE Background Document: [http://www.eggie5.com/134-nce-Noise-contrastive-Estimation-Loss](http://www.eggie5.com/134-nce-Noise-contrastive-Estimation-Loss)



```python
from keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
)
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

    sm_logits = NCE(num_users, name="nce")([h1, targets])

    model = Model(inputs=[iid, targets], outputs=[sm_logits])
    return model


K = 10
SAMPLE_SIZE = 10000
num_items = 10000
NUM_USERS = 1000000 #THIS IS SIZE OF SOFTMAX

model = build(num_items, NUM_USERS, K)
model.compile(optimizer="adam", loss=None)
model.summary()

x = np.random.random_integers(num_items - 1, size=SAMPLE_SIZE)
y = np.ones(SAMPLE_SIZE)
X = [x, y]
print(x.shape, y.shape)

model.fit(x=X, batch_size=100, epochs=1)

```

