import tensorflow as tf
from tensorflow import keras

def training(model, x_train, y_train, x_test=None, y_test=None, epochs=1):
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs)

    return model