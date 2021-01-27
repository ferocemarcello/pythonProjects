import tensorflow as tf
import arff
import numpy as np
import random

def getAnnotationMatrix(path):#gets the annotation matrix as sparse matrix
    decoder = arff.ArffDecoder()
    f = open(path, 'r')
    matrix = decoder.decode(f, encode_nominal=True, return_type=arff.LOD)
    f.close()
    matrix['attributes'].pop(0)#remove the first attribute, because it's "gene STRING"
    return matrix


def getTermsAnnotationMatrix(arfffile):#get all the terms
    terms = arfffile['attributes']
    return terms

def getCommonTerms(*termLists):  # gets the terms which are included in all the termlist in input
    cTerms = []
    shortestIndex = 0
    shortestLen = len(termLists[0])
    for termlist in termLists:
        if len(termlist) < shortestLen:
            shortestLen = len(termlist)
            shortestIndex = termLists.index(termlist)
    for t in termLists[shortestIndex]:
        rightIn = True
        for termlist in termLists:
            if t not in termlist:
                rightIn = False
                break
        if rightIn == True:
            cTerms.append(t)
    if len(cTerms) == 0:
        return None
    else:
        return cTerms


def getTermsLength(terms):  # gets the length of a list/array of terms
    return len(terms)

def getAllGenes(annotationMatrix):#gets all the gene names from a matrix
    geneList = []
    for gene in annotationMatrix['data']:
        geneList.append(gene[0])
    return geneList

def getPositiveAnnotIndexes(geneannotations):#gets all the indexes of the annotations of a gene, where the annotation is 1
    indexes = []
    for annot in geneannotations:
        indexes.append(annot)
    indexes.pop(0)
    return indexes

def getnumones(commonTerms, annotationsMatrix, geneannotations):#gets the number of annotations that are 1 for a list of annotations matching a list of common terms
    numones = 0
    for annot in geneannotations:
        if (annotationsMatrix['attributes'])[annot - 1] in commonTerms:
            numones = numones + 1
    return numones

def getGenesList(commonTerms, annotationMatrix, M, checkExistastanceinMatrix):  # gets all the terms from a version, included in the ones given in input, and whose annotations have at least M 1 ones
    geneList = []
    comparingGeneNames = getAllGenes(checkExistastanceinMatrix)
    for gene in annotationMatrix['data']:
        if gene[0] in comparingGeneNames:
            numones = getnumones(commonTerms, annotationMatrix, gene)
            if numones >= M:
                geneList.append(gene[0])
    return geneList
def getAncestorList(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    ancestorList = []
    for row in content:
        anc = row.split()
        ancestorList.append(anc)
    return ancestorList

def commontermsIndexes(commonterms, annotationMatrix):#get the indexes for the common terms in the annotation matrix
    commontermsIndexes = []
    for term in annotationMatrix['attributes']:
        if term in commonterms:
            commontermsIndexes.append(annotationMatrix['attributes'].index(term))
    return commontermsIndexes

def getAnnotations(gene, annotationMatrix, commonterms):#get all the annotations for the common terms in the annotation matrix
    annots = []
    annIndex = getAllGenes(annotationMatrix).index(gene)
    indexes = getPositiveAnnotIndexes((annotationMatrix['data'])[annIndex])
    for term in commonterms:
        ind = annotationMatrix['attributes'].index(term)
        if ind not in indexes:
            annots.append(0)
        else:
            annots.append(1)
    return annots

def getRandomAnnotations(genesList, annotationsMatrixA, annotationsMatrixB, batchsize, commonterms):  # gets the random annotations from the two versions given in input, matching the given terms, for a set of batchsize random(but included in both versions) genes
    selectedGenes = random.sample(genesList, batchsize)
    selectedAnnotationsA = []
    selectedAnnotationsB = []
    for gene in selectedGenes:
        annotationsA = getAnnotations(gene, annotationsMatrixA, commonterms)
        annotationsB = getAnnotations(gene, annotationsMatrixB, commonterms)
        selectedAnnotationsA.append(annotationsA)
        selectedAnnotationsB.append(annotationsB)
    return selectedAnnotationsA, selectedAnnotationsB

def getCommonGenes(annotationsTargetOld, annotationsTargetNew):#gets the list of genes which are included in both the annotations in input
    genesA = getAllGenes(annotationsTargetOld)
    genesB = getAllGenes(annotationsTargetNew)
    commonGenes = []
    if len(genesA) < len(genesB):
        for gene in genesA:
            if gene in genesB:
                commonGenes.append(gene)
    else:
        for gene in genesB:
            if gene in genesA:
                commonGenes.append(gene)
    return commonGenes
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def getFirstColumnFromMultiColumnList(list):
    col=[]
    for row in list:
        col.append(row[0])
    return col
def correctLikelihoodAnn(ann,termindex,annList,ancestorList,commonTerms):
    term=commonTerms[termindex]
    if term in getFirstColumnFromMultiColumnList(ancestorList):
        termlistindex=((np.asarray(ancestorList)[:,0]).tolist()).index(term)
        sumhierlikelihood=0
        numancestors=0
        for i in range(len(ancestorList[termlistindex])):
            if i!=0:
                try:
                    sumhierlikelihood=sumhierlikelihood+annList[commonTerms.index(ancestorList[termlistindex][i])]
                    numancestors=numancestors+1
                except Exception as e:
                    print(e)
        ann=(sumhierlikelihood/numancestors + ann)/2
def correctLikelihoodMatrix(onlyAnnotMatrix, ancestorList, commonTerms):
    for annList in onlyAnnotMatrix:
        for ann in annList:
            correctLikelihoodAnn(ann,annList.tolist().index(ann),annList,ancestorList,commonTerms)
def getOneColumnFromMatrix(matrix):
    col=[]
    for row in matrix:
        for ann in row:
            col.append(ann)
    return np.asarray(col)
M = 30
pathAold = '/home/feroce/Scrivania/gendata/unfolded/mm_2009_unfolded.arff'  # old version species A, A source organism. It should be a well know organism
pathBold = '/home/feroce/Scrivania/gendata/unfolded/bt_2009_unfolded.arff'  # old version species B, B target organism. It makes sense that B were a bad know organism
pathAnew = '/home/feroce/Scrivania/gendata/unfolded/mm_2013_unfolded.arff'  # recent version species A, A source organism. It should be a well know organism
pathBnew = '/home/feroce/Scrivania/gendata/unfolded/bt_2013_unfolded.arff'  # recent version species B, B target organism. It makes sense that B were a bad know organism. this should be used just for testing
pathAncestorOld = '/home/feroce/Scrivania/gendata/ancestors_2009.txt'
pathAncestorNew = '/home/feroce/Scrivania/gendata/ancestors_2013.txt'
ancestorListOld = getAncestorList(pathAncestorOld)
ancestorListNew = getAncestorList(pathAncestorNew)
annotationsTargetOld=getAnnotationMatrix(pathBold)
print('first matrix')

annotationsSourceNew = getAnnotationMatrix(pathAnew)
print('second matrix')

annotationsSourceOld = getAnnotationMatrix(pathAold)
print('third matrix')

annotationsTargetNew = getAnnotationMatrix(pathBnew)
print('fourth matrix')

termsA = getTermsAnnotationMatrix(annotationsSourceOld)
termsB = getTermsAnnotationMatrix(annotationsTargetOld)
termsAnew = getTermsAnnotationMatrix(annotationsSourceNew)
termsBnew = getTermsAnnotationMatrix(annotationsTargetNew)

commonTerms = getCommonTerms(termsA, termsB, termsAnew, termsBnew)#I have to pass also the recent annotation matrixes, because some terms may will not be present in these ones, although they are more recent
genesList = getGenesList(commonTerms, annotationsSourceOld, M, annotationsSourceNew)#same speech for the the genes, some genes in the old matrix for the source, may will not be present in the recent one, so I am passing also the recent matrix
termlength = getTermsLength(commonTerms)
print('termlength: ', termlength)

x = tf.placeholder(tf.float32, [None,1])  # each input is a list of annotations for a gene in a version. So it has to have lenght corresponding the number of the terms that are considered
W1 = tf.Variable(tf.random_normal([1, 1], stddev=0.35))
b1 = tf.Variable(tf.zeros([1]))  # you can see the output has the same size and format of the input
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)# predicted annotations

W2 = tf.Variable(tf.random_normal([1, 1], stddev=0.35))
b2 = tf.Variable(tf.zeros([1]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(tf.random_normal([1, 1], stddev=0.35))
b3 = tf.Variable(tf.zeros([1]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

y=y3
print('ended declaration of parameters')
y_ = tf.placeholder(tf.float32, [None,1])  # true annotations
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))  # function for the error
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)  # minimize the error
out=tf.round(y)
batchsize = 100# number of genes given per iteration
iterations = 100
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('starting of proper training')
for _ in range(iterations):
    print(_)#just print, for seeing the progress
    batch_xs, batch_ys = getRandomAnnotations(genesList, annotationsSourceOld, annotationsSourceNew, batchsize,commonTerms)  # gets the random annotations for the source and target version(annotation matrix), for a certain number of random genes
    arr_xs=np.asarray(batch_xs)
    arr_ys = np.asarray(batch_ys)
    for i in range(len(arr_xs)):
        sess.run(train_step, feed_dict={x: sess.run(tf.reshape(arr_xs[i],[termlength,1])), y_: sess.run(tf.reshape(arr_ys[i],[termlength,1]))})  # this operation should modify the weights and the biases, and also the predicted annotations y

correct_prediction = tf.equal(out, y_)#as i told, here tf.equal gives a list of booleans, each one is true if the list of annotations for each gene is true to its mate
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # gets the value of the accuracy, casting booelans to integers, and then doing the mean
annotationsTargetOldannots = []
annotationsTargetNewannots = []
commonGenes = getCommonGenes(annotationsTargetOld, annotationsTargetNew)

for gene in commonGenes:
    annotationsTargetOldannots.append(getAnnotations(gene, annotationsTargetOld, commonTerms))
    annotationsTargetNewannots.append(getAnnotations(gene, annotationsTargetNew, commonTerms))

#print(sess.run(accuracy,feed_dict={x: np.asarray(annotationsTargetOldannots), y_: np.asarray(annotationsTargetNewannots)}))  # print the accuracy
annotationsTargetOldannotsInOneColumn=getOneColumnFromMatrix(np.asarray(annotationsTargetOldannots))
annotationsTargetOldannotsInOneColumn=annotationsTargetOldannotsInOneColumn.reshape(annotationsTargetOldannotsInOneColumn.shape[0],1)
annotationsTargetNewannotsInOneColumn=getOneColumnFromMatrix(np.asarray(annotationsTargetNewannots))
annotationsTargetNewnnotsInOneColumn=annotationsTargetNewannotsInOneColumn.reshape(annotationsTargetNewannotsInOneColumn.shape[0],1)

sigm=sess.run(y,feed_dict={x: annotationsTargetOldannotsInOneColumn})
countthreshold=0
countthresholdconfirmed=0
for z in range(len(sigm)):
    if sigm[z] <= 0.2:
        countthreshold = countthreshold + 1
        if annotationsTargetNewannots[z] == 0:
            countthresholdconfirmed = countthresholdconfirmed + 1
    if sigm[z] >= 0.8:
        countthreshold = countthreshold + 1
        if annotationsTargetNewannots[z] == 1:
            countthresholdconfirmed = countthresholdconfirmed + 1
prec=countthresholdconfirmed/countthreshold

'''correctLikelihoodMatrix(sigm, ancestorListNew,commonTerms)
rr=sess.run(out,feed_dict={x: np.asarray(annotationsTargetOldannots), y_: np.asarray(annotationsTargetNewannots)})
correct_prediction = tf.equal(annotationsTargetOldannots, annotationsTargetNewannots)
print(sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32))))'''

