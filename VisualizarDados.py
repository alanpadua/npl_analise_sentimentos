from typing import Dict
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import tokenize
from wordcloud import WordCloud


class VisualizarDados:
    def __init__(self) -> None:
        pass

    def visualizar_word_cloud(self, nuvem_palavras: WordCloud):
        plt.figure(figsize=(6, 7))
        plt.imshow(nuvem_palavras, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def pareto(self, dados, coluna_texto, quantidade):
        token_espaco = tokenize.WhitespaceTokenizer()

        todas_palavras = ' '.join([texto for texto in dados[coluna_texto]])
        token_frase = token_espaco.tokenize(todas_palavras)
        frequencia = nltk.FreqDist(token_frase)
        df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                      "Frequência": list(frequencia.values())})
        df_frequencia = df_frequencia.nlargest(columns="Frequência", n=quantidade)
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=df_frequencia, x="Palavra", y="Frequência", color='gray')
        ax.set(ylabel="Contagem")
        plt.show()

    def visualizar_accuracia(self, epochs_x, linhas: list, titulos:Dict):
        
        for x in linhas:
            plt.plot(epochs_x, x[0], x[1], label=x[2])

        plt.title(titulos['title'])
        plt.xlabel(titulos['xlabel'])
        plt.ylabel(titulos['ylabel'])
        plt.legend()

        plt.show()

    def valor_accuracia(self, history_dict):
        acc         = history_dict['accuracy']
        val_acc     = history_dict['val_accuracy']
        loss        = history_dict['loss']
        val_loss    = history_dict['val_loss']
        return loss, val_loss, acc, val_acc
