import nltk

from db import db_connection,db_operator
from nltk.corpus import comtrans
from nltk.translate import AlignedSent, Alignment
from nltk.translate import IBMModel1
class align:
    @staticmethod
    def do():
        # words = comtrans.words('alignment-en-fr.txt')
        db = db_connection()
        selector = db_operator(db)
        db.connect()
        query = 'Select * from reviews limit 100'
        sel = selector.executeSelection(query=query)
        db.disconnect()
        words = comtrans.words('alignment-en-fr.txt')
        als = comtrans.aligned_sents('alignment-en-fr.txt')[0]
        '''
        words (list(str)) – Words in the target language sentence
        mots (list(str)) – Words in the source language sentence
        alignment (Alignment) – Word-level alignments between words and mots. Each alignment is represented as a 2-tuple (words_index, mots_index).
        '''
        print(" ".join(als.words))
        print(" ".join(als.mots))
        print(als.alignment)
        print(als.invert())#to invert
        als = AlignedSent(['Reprise', 'de', 'la', 'session'],['Resumption', 'of', 'the', 'session'],Alignment([(0, 0), (1, 3), (2, 1), (3, 3)]))
        als.alignment = Alignment([(0, 0), (1, 1), (2, 2, "boat"), (3, 3, False, (1, 2))])


        corpus = [AlignedSent(['the', 'house'], ['das', 'Haus']),AlignedSent(['the', 'book'], ['das', 'Buch']),
                  AlignedSent(['a', 'book'], ['ein', 'Buch'])]
        em_ibm1 = IBMModel1(corpus, 20)
        com_ibm1 = IBMModel1(comtrans.aligned_sents()[:10], 20)
        print(com_ibm1.prob_alignment_point('bitte','Please'))

        my_als = AlignedSent(['Resumption', 'of', 'the', 'session'],['Reprise', 'de', 'la', 'session'],alignment=Alignment([(0, 0), (3, 3), (1, 2), (1, 1), (1, 3)]))

        #precision is simply interested in the proportion of correct alignments
        print(nltk.precision({0,1,2,3}, {0,1}))#1
        print(nltk.precision({0,1,2,3}, {0,5}))#0.5
        print(nltk.precision({0, 1, 2, 3}, {6, 5}))  # 0
        #recall is simply interested in the proportion of found alignments,
        print(nltk.recall({0, 1, 2, 3}, {0, 1}))  # 0.5
        print(nltk.recall({0, 1, 2, 3}, {0, 5}))  # 0.25
        print(nltk.recall({0, 1, 2, 3}, {6, 5}))  # 0
        #AER = 1 - (|A∩S| + |A∩P|) / (|A| + |S|) s=sure alignments, p=possible alignments, a=test alignments
        #Alignment Error Rate is commonly used metric for assessing sentence alignments. It combines precision and recall metrics together such that a perfect alignment must have all of the sure alignments and may have some possible alignments
        # Return an error rate between 0.0(perfect alignment) and 1.0(no alignment).
        print(nltk.alignment_error_rate(Alignment([(0, 0), (1, 1), (2, 2)]),Alignment([(0, 0), (1, 2), (2, 1)])))#0.6666666666666667
        return sel