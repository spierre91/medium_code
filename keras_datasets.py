"""
Created on Fri Feb 14 15:30:22 2020

@author: sadrachpierre
"""

from tensorflow.keras.datasets import cifar10  
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt 

plt.imshow(X_train[0])

plt.show()



from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')





from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
