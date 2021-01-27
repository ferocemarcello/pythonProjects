from __future__ import unicode_literals, print_function, division
from db import db_connection, db_operator
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from nltk.corpus import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

#https://www.guru99.com/nltk-tutorial.html
class guru99:
    @staticmethod
    def do():
        db = db_connection()
        selector = db_operator(db)
        db.connect()
        query = 'Select * from reviews limit 100'
        sel = selector.executeSelection(query=query)
        db.disconnect()

        #ŧagging and chunking
        text = "learn php from guru99"
        tokens = nltk.word_tokenize(text)
        print(tokens)
        tag = nltk.pos_tag(tokens)
        print(tag)
        grammar = "NP: {<DT>?<JJ>*<NN>}"
        '''
        .	Any character except new line
        *	Match 0 or more repetitions
        ?	Match 0 or 1 repetitions
        '''
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(tag)
        print(result)
        result.draw()  # It will draw the pattern graphically which can be seen in Noun Phrase chunking

        #Finding Synonyms
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets("active"):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

        print(set(synonyms))
        print(set(antonyms))

        #Collocations: Bigrams and Trigrams
        text = "Guru99 is a totally new kind of learning experience."
        Tokens = nltk.word_tokenize(text)
        output = list(nltk.bigrams(Tokens))
        print(output)
        output = list(nltk.trigrams(Tokens))
        print(output)

        #word2vec
        vectorizer = CountVectorizer()
        data_corpus = ["guru99 is the best sitefor online tutorials. I love to visit guru99."]
        vocabulary = vectorizer.fit(data_corpus)
        X = vectorizer.transform(data_corpus)
        print(X.toarray())
        print(vocabulary.get_feature_names())

        #Relation of NLTK and Word2vec
        model = gensim.models.Word2Vec(abc.sents())
        X = list(model.wv.vocab)
        data = model.most_similar('science')
        print(data)

        # list of libraries used by the code
        import string
        from gensim.models import Word2Vec
        import logging
        from nltk.corpus import stopwords
        from textblob import Word
        import json
        import pandas as pd
        # data in json format
        json_file = 'intents.json'
        with open('intents.json', 'r') as f:
            data = json.load(f)
        # displaying the list of stopwords
        stop = stopwords.words('english')
        # dataframe
        df = pd.DataFrame(data)

        df['patterns'] = df['patterns'].apply(', '.join)
        # print(df['patterns'])
        # print(df['patterns'])
        # cleaning the data using the NLP approach
        print(df)
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
        df['patterns'] = df['patterns'].str.replace('[^\w\s]', '')
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if not x.isdigit()))
        df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if not x in stop))
        df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        # taking the outer list
        bigger_list = []
        for i in df['patterns']:
            li = list(i.split(" "))
            bigger_list.append(li)
        # structure of data to be taken by the model.word2vec
        print("Data format for the overall list:", bigger_list)
        # custom data is fed to machine for further processing
        model = Word2Vec(bigger_list, min_count=1, size=300, workers=4)
        # print(model)
        model.save("word2vec.model")
        model.save("model.bin")
        model = Word2Vec.load('model.bin')
        vocab = list(model.wv.vocab)
        similar_words = model.most_similar('thanks')
        print(similar_words)
        dissimlar_words = model.doesnt_match('See you later, thanks for visiting'.split())
        print(dissimlar_words)
        similarity_two_words = model.similarity('please', 'see')
        print("Please provide the similarity between these two words:")
        print(similarity_two_words)
        similar = model.similar_by_word('kind')
        print(similar)


        ###
        ###seq2seq
        ###

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        SOS_token = 0
        EOS_token = 1
        MAX_LENGTH = 20

        # initialize Lang Class
        class Lang:
            def __init__(self):
                # initialize containers to hold the words and corresponding index
                self.word2index = {}
                self.word2count = {}
                self.index2word = {0: "SOS", 1: "EOS"}
                self.n_words = 2  # Count SOS and EOS

            # split a sentence into words and add it to the container
            def addSentence(self, sentence):
                for word in sentence.split(' '):
                    self.addWord(word)

            # If the word is not in the container, the word will be added to it,
            # else, update the word counter
            def addWord(self, word):
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.word2count[word] = 1
                    self.index2word[self.n_words] = word
                    self.n_words += 1
                else:
                    self.word2count[word] += 1

        # Normalize every sentence
        def normalize_sentence(df, lang):
            sentence = df[lang].str.lower()
            sentence = sentence.str.replace('[^A-Za-z\s]+', '')
            sentence = sentence.str.normalize('NFD')
            sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
            return sentence

        def read_sentence(df, lang1, lang2):
            sentence1 = normalize_sentence(df, lang1)
            sentence2 = normalize_sentence(df, lang2)
            return sentence1, sentence2

        def read_file(loc, lang1, lang2):
            df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
            return df

        def process_data(lang1, lang2):
            df = read_file('text/%s-%s.txt' % (lang1, lang2), lang1, lang2)
            print("Read %s sentence pairs" % len(df))
            sentence1, sentence2 = read_sentence(df, lang1, lang2)

            source = Lang()
            target = Lang()
            pairs = []
            for i in range(len(df)):
                if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
                    full = [sentence1[i], sentence2[i]]
                    source.addSentence(sentence1[i])
                    target.addSentence(sentence2[i])
                    pairs.append(full)

            return source, target, pairs

        def indexesFromSentence(lang, sentence):
            return [lang.word2index[word] for word in sentence.split(' ')]

        def tensorFromSentence(lang, sentence):
            indexes = indexesFromSentence(lang, sentence)
            indexes.append(EOS_token)
            return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

        def tensorsFromPair(input_lang, output_lang, pair):
            input_tensor = tensorFromSentence(input_lang, pair[0])
            target_tensor = tensorFromSentence(output_lang, pair[1])
            return (input_tensor, target_tensor)

        class Encoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
                super(Encoder, self).__init__()

                # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
                self.input_dim = input_dim
                self.embbed_dim = embbed_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                # initialize the embedding layer with input and embbed dimention
                self.embedding = nn.Embedding(input_dim, self.embbed_dim)
                # intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
                # set the number of gru layers
                self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

            def forward(self, src):
                embedded = self.embedding(src).view(1, 1, -1)
                outputs, hidden = self.gru(embedded)
                return outputs, hidden

        class Decoder(nn.Module):
            def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
                super(Decoder, self).__init__()

                # set the encoder output dimension, embed dimension, hidden dimension, and number of layers
                self.embbed_dim = embbed_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.num_layers = num_layers

                # initialize every layer with the appropriate dimension. For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
                self.embedding = nn.Embedding(output_dim, self.embbed_dim)
                self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
                self.out = nn.Linear(self.hidden_dim, output_dim)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, input, hidden):
                # reshape the input to (1, batch_size)
                input = input.view(1, -1)
                embedded = F.relu(self.embedding(input))
                output, hidden = self.gru(embedded, hidden)
                prediction = self.softmax(self.out(output[0]))

                return prediction, hidden

        class Seq2Seq(nn.Module):
            def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
                super().__init__()

                # initialize the encoder and decoder
                self.encoder = encoder
                self.decoder = decoder
                self.device = device

            def forward(self, source, target, teacher_forcing_ratio=0.5):

                input_length = source.size(0)  # get the input length (number of words in sentence)
                batch_size = target.shape[1]
                target_length = target.shape[0]
                vocab_size = self.decoder.output_dim

                # initialize a variable to hold the predicted outputs
                outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

                # encode every word in a sentence
                for i in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(source[i])

                # use the encoder’s hidden layer as the decoder hidden
                decoder_hidden = encoder_hidden.to(device)

                # add a token before the first predicted word
                decoder_input = torch.tensor([SOS_token], device=device)  # SOS

                # topk is used to get the top K value over a list
                # predict the output word from the current target word. If we enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output highest value.

                for t in range(target_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    teacher_force = random.random() < teacher_forcing_ratio
                    topv, topi = decoder_output.topk(1)
                    input = (target[t] if teacher_force else topi)
                    if (teacher_force == False and input.item() == EOS_token):
                        break

                return outputs

        teacher_forcing_ratio = 0.5

        def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
            model_optimizer.zero_grad()

            input_length = input_tensor.size(0)
            loss = 0
            epoch_loss = 0
            # print(input_tensor.shape)

            output = model(input_tensor, target_tensor)

            num_iter = output.size(0)
            print(num_iter)

            # calculate the loss from a predicted sentence with the expected result
            for ot in range(num_iter):
                loss += criterion(output[ot], target_tensor[ot])

            loss.backward()
            model_optimizer.step()
            epoch_loss = loss.item() / num_iter

            return epoch_loss

        def trainModel(model, source, target, pairs, num_iteration=20000):
            model.train()

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.NLLLoss()
            total_loss_iterations = 0

            training_pairs = [tensorsFromPair(source, target, random.choice(pairs))
                              for i in range(num_iteration)]

            for iter in range(1, num_iteration + 1):
                training_pair = training_pairs[iter - 1]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                loss = clacModel(model, input_tensor, target_tensor, optimizer, criterion)

                total_loss_iterations += loss

                if iter % 5000 == 0:
                    avarage_loss = total_loss_iterations / 5000
                    total_loss_iterations = 0
                    print('%d %.4f' % (iter, avarage_loss))

            torch.save(model.state_dict(), 'mytraining.pt')
            return model

        def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
            with torch.no_grad():
                input_tensor = tensorFromSentence(input_lang, sentences[0])
                output_tensor = tensorFromSentence(output_lang, sentences[1])

                decoded_words = []

                output = model(input_tensor, output_tensor)
                # print(output_tensor)

                for ot in range(output.size(0)):
                    topv, topi = output[ot].topk(1)
                    # print(topi)

                    if topi[0].item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(output_lang.index2word[topi[0].item()])
            return decoded_words

        def evaluateRandomly(model, source, target, pairs, n=10):
            for i in range(n):
                pair = random.choice(pairs)
                print('source{}'.format(pair[0]))
                print('target{}'.format(pair[1]))
                output_words = evaluate(model, source, target, pair)
                output_sentence = ' '.join(output_words)
                print('predicted{}'.format(output_sentence))

        lang1 = 'eng'
        lang2 = 'ind'
        source, target, pairs = process_data(lang1, lang2)

        randomize = random.choice(pairs)
        print('random sentence {}'.format(randomize))

        # print number of words
        input_size = source.n_words
        output_size = target.n_words
        print('Input : {} Output : {}'.format(input_size, output_size))

        embed_size = 256
        hidden_size = 512
        num_layers = 1
        num_iteration = 100000

        # create encoder-decoder model
        encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
        decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

        model = Seq2Seq(encoder, decoder, device).to(device)

        # print model
        print(encoder)
        print(decoder)

        model = trainModel(model, source, target, pairs, num_iteration)
        evaluateRandomly(model, source, target, pairs)
        return None