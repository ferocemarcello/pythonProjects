import nltk

from db import db_connection, db_operator
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import names

#https://pythonspot.com/category/nltk/
class pythonspot:
    @staticmethod
    def do():
        db = db_connection()
        selector = db_operator(db)
        db.connect()
        query = 'Select * from reviews limit 100'
        sel = selector.executeSelection(query=query)
        db.disconnect()

        data = "All work and no play makes jack a dull boy, all work and no play"
        print(word_tokenize(data))#Tokenize words
        data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
        print(sent_tokenize(data))#Tokenize sentences

        #NLTK and arrays
        data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
        phrases = sent_tokenize(data)
        words = word_tokenize(data)
        print(phrases)
        print(words)

        #Natural Language Processing: remove stop words
        data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(data)
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        print(wordsFiltered)

        #NLTK – stemming
        words = ["game", "gaming", "gamed", "games"]
        ps = PorterStemmer()
        for word in words:
            print(ps.stem(word))
        sentence = "gaming, the gamers play games"
        words = word_tokenize(sentence)
        for word in words:
            print(word + ":" + ps.stem(word))

        #NLTK speech tagging
        document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
        sentences = nltk.sent_tokenize(document)
        for sent in sentences:
            print(nltk.pos_tag(nltk.word_tokenize(sent)))

        document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
        sentences = nltk.sent_tokenize(document)
        data = []
        for sent in sentences:
            data = data + nltk.pos_tag(nltk.word_tokenize(sent))
        for word in data:
            if 'NNP' in word[1]:
                print(word)

        #Natural Language Processing – prediction
        # Load data and training
        nltk.corpus.names
        '''names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])'''#not updated

        #Sentiment Analysis
        def word_feats(words):
            return dict([(word, True) for word in words])

        positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
        negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
        neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']

        positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
        negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
        neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

        train_set = negative_features + positive_features + neutral_features

        classifier = nltk.NaiveBayesClassifier.train(train_set)

        # Predict
        neg = 0
        pos = 0
        sentence = "Awesome movie, I liked it"
        sentence = sentence.lower()
        words = sentence.split(' ')
        for word in words:
            classResult = classifier.classify(word_feats(word))
            if classResult == 'neg':
                neg = neg + 1
            if classResult == 'pos':
                pos = pos + 1

        print('Positive: ' + str(float(pos) / len(words)))
        print('Negative: ' + str(float(neg) / len(words)))
        return None