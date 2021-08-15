
# warnings.filterwarnings('ignore')
import pickle
import re
import string
import warnings

import nltk
import pandas as pd
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas.core.construction import array
from pandas.io.formats.format import return_docstring
from wordcloud import WordCloud


class TratarDados:
    def __init__(self):
        self.arquivo: str = "dados/training.1600000.processed.noemoticon.csv"
        self.tweets: pd.DataFrame

        self.REPLACE_NO_SPACE = re.compile(
            "(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\]|(\@))")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        pass

    def preprocess_reviews(self, dados: pd.DataFrame):
        dados['Texto'] = dados['Texto'].apply(
            lambda row: self.REPLACE_NO_SPACE.sub(" ", row))
        dados['Texto'] = dados['Texto'].apply(
            lambda row: self.REPLACE_WITH_SPACE.sub(" ", row))

        return dados

    def importar_csv(self):
        tweets = pd.read_csv(self.arquivo, encoding='latin-1',
                             names=['Sentimento', 'Index', 'Data', 'Tipo', 'Usuario', 'Texto'])
        tweets = pd.DataFrame(
            tweets, columns=['Sentimento', 'Data', 'Usuario', 'Texto'])

        return tweets

    def remover_stop_words(self,  dados: pd.DataFrame, coluna: str):
        token_espaco = tokenize.WhitespaceTokenizer()
        palavras_irrelevantes = nltk.corpus.stopwords.words("english")

        frase_processada = list()
        for opiniao in dados[coluna]:
            nova_frase = list()
            palavras_texto = token_espaco.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in palavras_irrelevantes:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))

        return frase_processada

    def contar_palavras(self, dados: pd.DataFrame):
        token_espaco = tokenize.WhitespaceTokenizer()

        count = 0
        for frase in dados.Texto:
            token = token_espaco.tokenize(frase)
            count += len(token)
            # print(token, count)

        # print(count)
        return count

    def importar_parcial(self, quantidade: int):
        tweets = self.importar_csv()
        # Reduzindo para ficar mais rápido
        # Será removido

        # tweets = tweets.iloc[0:1000]
        positivos = tweets.query("Sentimento == 4").iloc[0:quantidade]
        negativos = tweets.query("Sentimento == 0").iloc[0:quantidade]
        neutros = tweets.query("Sentimento == 2").iloc[0:quantidade]

        tweets = pd.DataFrame(positivos)
        tweets = tweets.append(negativos)
        tweets = tweets.append(neutros)

        return tweets

    def criar_nuvem_palavras(self, dados: pd.DataFrame, campo: str):
        todos_palavras = ' '.join([texto for texto in dados[campo]])
        nuvem_palavras = WordCloud(
            width=800, height=500,
            max_font_size=110,
            collocations=False).generate(todos_palavras)

        return nuvem_palavras

    def process_tweets(self, tweet):
        urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern = '@[^\s]+'
        stopword = set(stopwords.words('english'))

        some = 'amp|today|tomorrow|going|girl|tonight|getting|day|get|http|com|go|one|lol'

        # Lower Casing
        tweet = re.sub(r"he's", "he is", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"We're", "We are", tweet)
        tweet = re.sub(r"That's", "That is", tweet)
        tweet = re.sub(r"won't", "will not", tweet)
        tweet = re.sub(r"they're", "they are", tweet)
        tweet = re.sub(r"Can't", "Cannot", tweet)
        tweet = re.sub(r"wasn't", "was not", tweet)
        tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
        tweet = re.sub(r"aren't", "are not", tweet)
        tweet = re.sub(r"isn't", "is not", tweet)
        tweet = re.sub(r"What's", "What is", tweet)
        tweet = re.sub(r"haven't", "have not", tweet)
        tweet = re.sub(r"hasn't", "has not", tweet)
        tweet = re.sub(r"There's", "There is", tweet)
        tweet = re.sub(r"He's", "He is", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"You're", "You are", tweet)
        tweet = re.sub(r"I'M", "I am", tweet)
        tweet = re.sub(r"shouldn't", "should not", tweet)
        tweet = re.sub(r"wouldn't", "would not", tweet)
        tweet = re.sub(r"i'm", "I am", tweet)
        tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
        tweet = re.sub(r"I'm", "I am", tweet)
        tweet = re.sub(r"Isn't", "is not", tweet)
        tweet = re.sub(r"Here's", "Here is", tweet)
        tweet = re.sub(r"you've", "you have", tweet)
        tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
        tweet = re.sub(r"we're", "we are", tweet)
        tweet = re.sub(r"what's", "what is", tweet)
        tweet = re.sub(r"couldn't", "could not", tweet)
        tweet = re.sub(r"we've", "we have", tweet)
        tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
        tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
        tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
        tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
        tweet = re.sub(r"who's", "who is", tweet)
        tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
        tweet = re.sub(r"y'all", "you all", tweet)
        tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
        tweet = re.sub(r"would've", "would have", tweet)
        tweet = re.sub(r"it'll", "it will", tweet)
        tweet = re.sub(r"we'll", "we will", tweet)
        tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
        tweet = re.sub(r"We've", "We have", tweet)
        tweet = re.sub(r"he'll", "he will", tweet)
        tweet = re.sub(r"Y'all", "You all", tweet)
        tweet = re.sub(r"Weren't", "Were not", tweet)
        tweet = re.sub(r"Didn't", "Did not", tweet)
        tweet = re.sub(r"they'll", "they will", tweet)
        tweet = re.sub(r"they'd", "they would", tweet)
        tweet = re.sub(r"DON'T", "DO NOT", tweet)
        tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
        tweet = re.sub(r"they've", "they have", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"should've", "should have", tweet)
        tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
        tweet = re.sub(r"where's", "where is", tweet)
        tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
        tweet = re.sub(r"we'd", "we would", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"weren't", "were not", tweet)
        tweet = re.sub(r"They're", "They are", tweet)
        tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
        tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
        tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
        tweet = re.sub(r"let's", "let us", tweet)
        tweet = re.sub(r"it's", "it is", tweet)
        tweet = re.sub(r"can't", "cannot", tweet)
        tweet = re.sub(r"don't", "do not", tweet)
        tweet = re.sub(r"you're", "you are", tweet)
        tweet = re.sub(r"i've", "I have", tweet)
        tweet = re.sub(r"that's", "that is", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"doesn't", "does not", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"didn't", "did not", tweet)
        tweet = re.sub(r"ain't", "am not", tweet)
        tweet = re.sub(r"you'll", "you will", tweet)
        tweet = re.sub(r"I've", "I have", tweet)
        tweet = re.sub(r"Don't", "do not", tweet)
        tweet = re.sub(r"I'll", "I will", tweet)
        tweet = re.sub(r"I'd", "I would", tweet)
        tweet = re.sub(r"Let's", "Let us", tweet)
        tweet = re.sub(r"you'd", "You would", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"Ain't", "am not", tweet)
        tweet = re.sub(r"Haven't", "Have not", tweet)
        tweet = re.sub(r"Could've", "Could have", tweet)
        tweet = re.sub(r"youve", "you have", tweet)
        tweet = re.sub(r"donå«t", "do not", tweet)
        tweet = re.sub(r"some1", "someone", tweet)
        tweet = re.sub(r"yrs", "years", tweet)
        tweet = re.sub(r"hrs", "hours", tweet)
        tweet = re.sub(r"2morow|2moro", "tomorrow", tweet)
        tweet = re.sub(r"2day", "today", tweet)
        tweet = re.sub(r"4got|4gotten", "forget", tweet)
        tweet = re.sub(r"b-day|bday", "b-day", tweet)
        tweet = re.sub(r"mother's", "mother", tweet)
        tweet = re.sub(r"mom's", "mom", tweet)
        tweet = re.sub(r"dad's", "dad", tweet)
        tweet = re.sub(r"hahah|hahaha|hahahaha", "haha", tweet)
        tweet = re.sub(r"lmao|lolz|rofl", "lol", tweet)
        tweet = re.sub(r"thanx|thnx", "thanks", tweet)
        tweet = re.sub(r"goood", "good", tweet)
        tweet = re.sub(r"some1", "someone", tweet)
        tweet = re.sub(r"some1", "someone", tweet)
        tweet = tweet.lower()
        tweet = tweet[1:]

        # Remove todas URls
        tweet = re.sub(urlPattern, '', tweet)

        # Remove todos @username.
        tweet = re.sub(userPattern, '', tweet)

        # Remove some words
        tweet = re.sub(some, '', tweet)

        # Remove pontuações
        # tweet = tweet.translate(str.maketrans("", "", string.punctuation))

        # tokenizing words
        # tokens = word_tokenize(tweet)

        #tokens = [w for w in tokens if len(w)>2]
        # Remover Stop Words
        # final_tokens = [w for w in tokens if w not in stopword]

        # reducing a word to its word stem
        # wordLemm = WordNetLemmatizer()
        # finalwords = []
        # for w in final_tokens:
            # if len(w) > 1:
                # word = wordLemm.lemmatize(w)
                # finalwords.append(word)
        # return ' '.join(finalwords)
        return tweet
