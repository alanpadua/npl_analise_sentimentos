from tensorflow import keras
from tensorflow.keras import Sequential, backend, layers, models, optimizers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     InputLayer, MaxPooling2D)
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline


class DeepLearning:
    def __init__(self) -> None:
        pass

    def definir_modelo(self, shape):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=shape, name='Camada_Oculta_1'))
        model.add(layers.Dense(16, activation='relu', name='Camada_Oculta_2'))
        model.add(layers.Dense(1, activation='sigmoid', name='Saida'))
        # model.add(layers.Dense(1, activation='softmax', name='Saida'))

        return model

    def compilar_modelo(self, model: Sequential):
        # return model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
        #                      loss='binary_crossentropy',
        #                      metrics=['accuracy'])
         return model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

    def treinar_modelo(self, model, X_treino, y_treino, X_teste, y_teste, epochs=20):
        history = model.fit(X_treino,
                            y_treino,
                            epochs=20,
                            batch_size=512,
                            validation_data=(X_teste, y_teste))
        return history

    def roc_auc(self, predicao, y_teste):
        fpr, tpr, thresholds = metrics.roc_curve(y_teste, predicao)
        roc_auc = metrics.auc(fpr, tpr)

        return roc_auc