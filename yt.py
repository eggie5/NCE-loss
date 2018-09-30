from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Average, Reshape, Lambda, Activation
from keras.models import Model

import keras.backend as K

import numpy as np

def build(NUM_ITEMS, k):

    iid = Input(shape=(1,), dtype='int32', name='iids') 
    
    item_embedding = Embedding(input_dim=NUM_ITEMS, output_dim=k, input_length=1, name="item_embedding")
    selected_items = Flatten()(item_embedding(iid))

    h1 = Dense(k//2, activation="relu")(selected_items)
    sig = Dense(NUM_ITEMS, activation="softmax", name="softmax")(h1)

    model = Model(inputs=[iid], outputs=sig)
    return model


if __name__ == '__main__':
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    # from keras.utils import plot_model
    # plot_model(model, to_file="yt.png")    
    K=256
    BATCH_SIZE=64
    softmax_size=np.arange(2, 1100000, 100000)
    times=[]
    
    for num_items in softmax_size:
        print("\n\nSoftmax Width: ",num_items)
        model = build(num_items, K)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam") 
        model.summary()
        
        t0 = time.time()
        X=[np.random.rand(BATCH_SIZE,1)]
        y=np.ones(BATCH_SIZE)
        model.fit(x=X, y=y)
        times.append(time.time() - t0)
    
    print(softmax_size)
    print(times)
    plt.plot(softmax_size, times, '-x')
    plt.show()
    