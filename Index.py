from Treinamento import Treinamento
from TratarDados import TratarDados
import pandas as pd

class Index:
    def __init__(self) -> None:
        tratar_dados = TratarDados()
        treinamento  = Treinamento()
        tweets = tratar_dados.importar_parcial(100)
        tweets = tratar_dados.preprocess_reviews(tweets)
        
        # nuvem_palavras = tratar_dados.criar_nuvem_palavras(dados=tweets, campo="Texto")

        # print(tweets.head())
        # print(nuvem_palavras)

        # count_words = tratar_dados.contar_palavras(tweets)
        vetorizar, bag_of_words = treinamento.vetorizar(tweets, 'Texto')
        X_treino, X_teste, y_treino, y_teste= treinamento.separar_dados_treinamento(bag_of_words.toarray(), tweets.Sentimento)
        nb = treinamento.naive_bayes(X_treino, y_treino, X_teste, y_teste)

        # print(count_words)
        pass



index = Index()

    