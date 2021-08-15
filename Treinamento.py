from re import T

import numpy as np
import pandas as pd
from numpy.core.records import array
from pandas.io.stata import precision_loss_doc
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud.tokenization import score


class Treinamento:
    def __init__(self) -> None:
        pass

    def logistic_regression(self, X_treino, y_treino, X_teste, y_teste):
        logistic_regression = LogisticRegression(random_state=0)
        logistic_regression.fit(X_treino, y_treino)
        y_predicao = logistic_regression.predict(X_teste)

        return y_predicao
    
    def classificar_texto(self, tweets: pd.DataFrame, coluna_texto, coluna_classificacao):
        vetorizar, bag_of_words = self.vetorizar(tweets, coluna_texto)
        X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words,
                                                                tweets[coluna_classificacao],
                                                                random_state = 42)
        regressao_logistica = LogisticRegression(solver = "lbfgs")
        regressao_logistica.fit(X_treino, y_treino)
        
        return regressao_logistica.score(X_teste, y_teste)

    def separar_dados_treinamento(self, matriz_treino: array, resultado_matriz):
        X_treino, X_teste, y_treino, y_teste = train_test_split(matriz_treino, resultado_matriz, random_state=42, test_size=0.3)

        return X_treino, X_teste, y_treino, y_teste

    def vetorizar(self, tweets: pd.DataFrame, coluna_texto: str):
        vetorizar = CountVectorizer(lowercase=False, max_features=50, stop_words='english')
        bag_of_words = vetorizar.fit_transform(tweets[coluna_texto])

        return vetorizar, bag_of_words

    def criar_matriz_esparsa(self, bag_of_words, vetorizar: CountVectorizer):
        matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names())

        return matriz_esparsa

    def naive_bayes(self, X_treino, y_treino, X_teste, y_teste):
        nb = MultinomialNB()
        nb.fit(X_treino, y_treino)
        y_predicao = nb.predict(X_teste)

        return y_predicao

    def metricas(self, y_teste, y_predicao):
        #Check performance do seu  modelo com classification report
        classificacao = classification_report(y_teste, y_predicao)

        #Check performance do seu  modelo com ROC Score
        score = roc_auc_score(y_teste, y_predicao)

        return classificacao, score
