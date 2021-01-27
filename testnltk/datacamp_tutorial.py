import nltk

from db import db_connection, db_operator
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

#https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
class datacamp:
    @staticmethod
    def do():
        db = db_connection()
        selector = db_operator(db)
        db.connect()
        query = 'Select * from reviews limit 100'
        sel = selector.executeSelection(query=query)
        db.disconnect()
        text = """Hello Mr. Smith, how are you doing today? The weather is great, and the city is awesome.
        The sky is pinkish-blue. You shouldn't eat cardboard"""
        tokenized_text = sent_tokenize(text)
        print(tokenized_text)#sentence tokenization

        tokenized_word = word_tokenize(text)#word tokenization
        print(tokenized_word)

        fdist = FreqDist(tokenized_word)#Frequency Distribution
        print(fdist)
        print(fdist.most_common(2))

        #fdist.plot(30, cumulative=False)
        #plt.show()

        stop_words = set(stopwords.words("english"))#Stopwords
        print(stop_words)

        filtered_sent = []
        for w in tokenized_word:#Removing Stopwords
            if w not in stop_words:
                filtered_sent.append(w)
        print("Tokenized Sentence:", tokenized_word)
        print("Filterd Sentence:", filtered_sent)

        ps = PorterStemmer()#Stemming
        stemmed_words = []
        for w in filtered_sent:
            stemmed_words.append(ps.stem(w))
        print("Filtered Sentence:", filtered_sent)
        print("Stemmed Sentence:", stemmed_words)

        lem = WordNetLemmatizer()#Lexicon Normalization(lemmalization)
        stem = PorterStemmer()
        word = "flying"
        print("Lemmatized Word:", lem.lemmatize(word, "v"))
        print("Stemmed Word:", stem.stem(word))

        sent = "Albert Einstein was born in Ulm, Germany in 1879."#Part-of-Speech(POS) tagging
        tokens = nltk.word_tokenize(sent)
        print(tokens)
        print(nltk.pos_tag(tokens))

        #Sentiment Analysis using Text Classification
        data = pd.read_csv('train.tsv', sep='\t')
        #This data has 5 sentiment labels: 0 - negative 1 - somewhat negative 2 - neutral 3 - somewhat positive 4 - positive
        print(data.head())
        print(data.info())
        print(data.Sentiment.value_counts())
        Sentiment_count = data.groupby('Sentiment').count()
        plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
        plt.xlabel('Review Sentiments')
        plt.ylabel('Number of Review')
        #plt.show()

        #Feature Generation using Bag of Words
        # tokenizer to remove unwanted elements from out data like symbols and numbers
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
        text_counts = cv.fit_transform(data['Phrase'])
        print(text_counts)

        #Split train and test set
        X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.3, random_state=1)
        # Model Generation Using Multinomial Naive Bayes
        clf = MultinomialNB().fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))

        #Feature Generation using TF-IDF
        tf = TfidfVectorizer()
        text_tf = tf.fit_transform(data['Phrase'])
        # Split train and test set (TF-IDF)
        X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)
        # Model Generation Using Multinomial Naive Bayes
        clf = MultinomialNB().fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
        return None