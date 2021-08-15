from tensorflow import keras
from tensorflow.keras import Sequential, backend, layers, models, optimizers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     InputLayer, MaxPooling2D)


class DeepLearning:
    def __init__(self) -> None:
        pass

    def definir_modelo(self, shape):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=shape))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def compilar_modelo(self, model: Sequential):
        return model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

    def treinar_modelo(self, model, X_treino, y_treino, X_teste, y_teste, epochs=20):
        history = model.fit(X_treino,
                            y_treino,
                            epochs=20,
                            batch_size=512,
                            validation_data=(X_teste, y_teste))
        return history
