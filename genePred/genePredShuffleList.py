import tensorflow as tf
import arff
import numpy as np
import random
import copy
import sys

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
def getTrainingList(genesList, annotationsMatrixA, annotationsMatrixB, batchsize, commonterms):
    selectedGenes = random.sample(genesList, batchsize)
    selectedAnnotationsA = []
    selectedAnnotationsB = []
    trainingList=[]
    for gene in selectedGenes:
        annotationsA = getAnnotations(gene, annotationsMatrixA, commonterms)
        annotationsB = getAnnotations(gene, annotationsMatrixB, commonterms)
        selectedAnnotationsA.append(annotationsA)
        selectedAnnotationsB.append(annotationsB)
    trainingList.append(selectedAnnotationsA)
    trainingList.append(selectedAnnotationsB)
    return trainingList
def getCommonTerms(*termLists):  # gets the terms which are included in all the termlist in input
    cTerms = []
    shortestIndex = 0
    shortestLen = len(termLists[0])
    for termlist in termLists:#way to determine the shortest list of terms
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
def getCommonGenes(*annotationMatrices):#gets the list of genes which are included in both the annotations in input
    commonGenes = []
    shortestIndex = 0
    shortestLen = len((annotationMatrices[0]['data']))
    for matr in annotationMatrices:#way to determine the shortest list of terms
        if len(matr['data']) < shortestLen:
            shortestLen = len(matr['data'])
            shortestIndex = annotationMatrices.index(matr['data'])
    for gene in (annotationMatrices[shortestIndex])['data']:
        rightIn = True
        for matr in annotationMatrices:
            matrixGenes=getFirstColumnFromMultiColumnList(matr['data'])
            if gene[0] not in matrixGenes:
                rightIn = False
                break
        if rightIn == True:
            commonGenes.append(gene[0])
    if len(commonGenes) == 0:
        return None
    else:
        return commonGenes
def getFirstColumnFromMultiColumnList(list):
    col=[]
    for row in list:
        col.append(row[0])
    return col
def correctLikelihoodAnn(ann,termindex,annList,ancestorList,commonTerms):
    term=commonTerms[termindex]
    term=(str)(term)
    term = term.replace(" ", "")
    term=term[:16]+term[17:]
    term = term[:19] + term[20:]
    term=term.replace("(","")
    term =term.replace(")","")
    term =term.replace(",","")
    term =term.replace("'","")
    term =term.replace("[","")
    term =term.replace("]","")
    firstColumnAncestorList=getFirstColumnFromMultiColumnList(ancestorList)
    if term in firstColumnAncestorList:
        termlistindex=firstColumnAncestorList.index(term)
        sumhierlikelihood=0
        numancestors=0
        atLeastOneTerm=False
        for i in range(1,len(ancestorList[termlistindex])):
            tup=(ancestorList[termlistindex][i],['0','1'])
            if tup in commonTerms:
                annindex=commonTerms.index(tup)
                sumhierlikelihood = sumhierlikelihood + annList[annindex]
                numancestors = numancestors + 1
                atLeastOneTerm = True
        if atLeastOneTerm==True:
            newann = (sumhierlikelihood / numancestors + ann) / 2
        else:
            newann = copy.deepcopy(ann)
        return newann
def correctLikelihoodMatrix(onlyAnnotMatrix, ancestorList, commonTerms):
    for i in range(len(onlyAnnotMatrix)):
        newarrann=np.zeros(len(onlyAnnotMatrix[i]))
        for z in range(len(onlyAnnotMatrix[i])):
            newann=correctLikelihoodAnn(onlyAnnotMatrix[i][z],z,onlyAnnotMatrix[i],ancestorList,commonTerms)
            newarrann[z]=newann
        onlyAnnotMatrix[i]=newarrann
        return onlyAnnotMatrix
def printInformationsArguments():
    print("argument 0: path of old matrix source organism 1")
    print("argument 1: path of new matrix source organism 1")
    print("argument 2: path of old matrix source organism 2")
    print("argument 3: path of new matrix source organism 2")
    print("argument 4: path of old matrix target organism")
    print("argument 5: path of new matrix target organism")
    print("argument 6: path of new IEA matrix target organism")
    print("argument 7: path of ancestor file of same age than new matrices of sources and target")
    print("argument 8: path of the output file where to store the results of the accuracy")
    print("argument 9: path of the output file where to store the predicted annotation matrix")
    print("argument 10: M value. It's an integer number")
    print("argument 11: batch size. It's an integer number")
    print("argument 12: number of iterations. It's an integer number")
    print("argument 13: tolerance. It's a float number")
def evaluateAccuracy(annMatrix,evaluatingMatrix,evaluatingMatrixIea,tolerance):
    countthreshold = 0
    countthresholdconfirmed = 0
    for z in range(len(annMatrix)):
        for i in range(len(annMatrix[z])):
            if annMatrix[z][i] <= 1-tolerance:
                countthreshold = countthreshold + 1
                if evaluatingMatrix[z][i] == 0 or evaluatingMatrixIea[z][i]==0:
                    countthresholdconfirmed = countthresholdconfirmed + 1
            if annMatrix[z][i] >= tolerance:
                countthreshold = countthreshold + 1
                if evaluatingMatrix[z][i]== 1 or evaluatingMatrixIea[z][i]==1:
                    countthresholdconfirmed = countthresholdconfirmed + 1
    prec = countthresholdconfirmed / countthreshold
    return  prec
def writeOnFile(outputfile, mode, text):
    out_file = open(outputfile, mode)
    if mode=="a":
        out_file.write("\n"+text)
    out_file.close()
def writeMatrixOnFile(path,matrix):
    encoder = arff.ArffEncoder()
    string = encoder.encode(matrix)
    file = open(path, 'w')
    file.write(string)
    file.close()
def getArffMatrix(annMatrix,terms,genes):
    sigmdict = dict()
    attributes =terms
    attributes.insert(0, ("gene", "STRING"))
    sigmdict["attributes"] = attributes
    data = annMatrix.tolist()
    dataindexes = []
    diclist = []
    for i in range(len(data)):
        data[i].insert(0, genes[i])
    for i in range(len(data[0])):
        dataindexes.append(i)
    for i in range(len(data)):
        diccc = zip(dataindexes, data[i])
        diccc = dict(diccc)
        diclist.append(diccc)
    sigmdict["data"] = diclist
    sigmdict["description"] = ""
    sigmdict["relation"] = "GOMatrix_organism_Type[EsBP]"
    otherann = False
    i = 1
    rowlen = len(sigmdict["attributes"])
    for row in sigmdict["data"]:
        i = 1
        while i < rowlen:
            if row.get(i) > 0.5:
                row[i] = 1
            else:
                del row[i]
            i = i + 1
    return sigmdict
def main():

    try:
        if sys.argv[1:][0]=="-help":
            printInformationsArguments()
            return
        pathAold=sys.argv[1:][0]
        pathAnew=sys.argv[1:][1]
        pathAold2 = sys.argv[1:][2]
        pathAnew2 = sys.argv[1:][3]
        pathBold=sys.argv[1:][4]
        pathBnew=sys.argv[1:][5]
        pathBnewIea = sys.argv[1:][6]
        pathAncestor = sys.argv[1:][7]
        outputfileAccuracy = sys.argv[1:][8]
        outputfileMatrix = sys.argv[1:][9]
        try:
            M=(int)(sys.argv[1:][10])
        except ValueError:
            print("insert a proper integer >=0 number for M")
            return
        if M<0:
            print("insert a proper integer >=0 number for M")
        try:
            batchsize = (int)(sys.argv[1:][11])  # number of genes given per iteration
        except ValueError:
            print("insert a proper integer >0 number for batchsize")
            return
        if batchsize<=0:
            print("insert a proper integer >0 number for batchsize")
        try:
            iterations = (int)(sys.argv[1:][12])
        except ValueError:
            print("insert a proper integer >0 number for iterations")
            return
        if iterations<=0:
            print("insert a proper integer >0 number for iterations")
        try:
            tolerance = float(sys.argv[1:][13])
            if tolerance>1 or tolerance<=0.5:
                print("insert a proper float number between 0.5(excluded) and 1(included)")
                return
        except ValueError:
            print("insert a proper float number between 0.5(excluded) and 1(included)")
            return
    except:
        print("not valid arguments")
        return
    ancestorList = getAncestorList(pathAncestor)
    print("getting matrices")
    annotationsTargetOld = getAnnotationMatrix(pathBold)
    annotationsSourceNew = getAnnotationMatrix(pathAnew)
    annotationsSourceOld = getAnnotationMatrix(pathAold)
    annotationsTargetNew = getAnnotationMatrix(pathBnew)
    annotationsTargetNewIea = getAnnotationMatrix(pathBnewIea)
    annotationsSourceNew2 = getAnnotationMatrix(pathAnew2)
    annotationsSourceOld2 = getAnnotationMatrix(pathAold2)
    print("got matrices")
    termsA = getTermsAnnotationMatrix(annotationsSourceOld)
    termsB = getTermsAnnotationMatrix(annotationsTargetOld)
    termsAnew = getTermsAnnotationMatrix(annotationsSourceNew)
    termsBnew = getTermsAnnotationMatrix(annotationsTargetNew)
    termsBnewIea = getTermsAnnotationMatrix(annotationsTargetNewIea)

    termsA2 = getTermsAnnotationMatrix(annotationsSourceOld2)
    termsA2new = getTermsAnnotationMatrix(annotationsSourceNew2)

    commonTerms = getCommonTerms(termsA, termsB, termsAnew, termsBnew, termsA2new, termsA2,
                                 termsBnewIea)  # I have to pass also the recent annotation matrixes, because some terms may will not be present in these ones, although they are more recent
    termlength = getTermsLength(commonTerms)
    print('common terms are : ', termlength)

    x = tf.placeholder(tf.float32, [None,
                                    termlength])  # each input is a list of annotations for a gene in a version. So it has to have lenght corresponding the number of the terms that are considered
    W1 = tf.Variable(tf.random_normal([termlength, termlength], stddev=0.35))
    b1 = tf.Variable(tf.zeros([termlength]))  # you can see the output has the same size and format of the input
    y1 = tf.sigmoid(tf.matmul(x, W1) + b1)  # predicted annotations

    W2 = tf.Variable(tf.random_normal([termlength, termlength], stddev=0.35))
    b2 = tf.Variable(tf.zeros([termlength]))
    y2 = tf.sigmoid(tf.matmul(y1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([termlength, termlength], stddev=0.35))
    b3 = tf.Variable(tf.zeros([termlength]))
    y3 = tf.sigmoid(tf.matmul(y2, W3) + b3)

    y = y3

    y_ = tf.placeholder(tf.float32, [None, termlength])  # true annotations
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))  # function for the error
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)  # minimize the error
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    genesList = getGenesList(commonTerms, annotationsSourceOld, M,
                             annotationsSourceNew)  # same speech for the the genes, some genes in the old matrix for the source, may will not be present in the recent one, so I am passing also the recent matrix
    print('starting of training')
    trainingList = getTrainingList(genesList, annotationsSourceOld, annotationsSourceNew, batchsize, commonTerms)
    print("first training")
    for i in range(iterations):
        if i%100==0:
            print(i)# just print, for seeing the progress
        trainingList = sess.run(tf.random_shuffle(trainingList))
        batch_xs = trainingList[
            0]  # gets the random annotations for the source and target version(annotation matrix), for a certain number of random genes
        batch_ys = trainingList[1]
        arr_xs = np.asarray(batch_xs)
        arr_ys = np.asarray(batch_ys)
        sess.run(train_step, feed_dict={x: arr_xs,
                                        y_: arr_ys})  # this operation should modify the weights and the biases, and also the predicted annotations y

    genesList = getGenesList(commonTerms, annotationsSourceOld2, M,
                             annotationsSourceNew2) # same speech for the the genes, some genes in the old matrix for the source, may will not be present in the recent one, so I am passing also the recent matrix

    trainingList = getTrainingList(genesList, annotationsSourceOld2, annotationsSourceNew2, batchsize, commonTerms)
    print("second training")
    for i in range(iterations):
        if i%100==0:
            print(i)# just print, for seeing the progress
        trainingList = sess.run(tf.random_shuffle(trainingList))
        batch_xs = trainingList[
            0]  # gets the random annotations for the source and target version(annotation matrix), for a certain number of random genes
        batch_ys = trainingList[1]
        arr_xs = np.asarray(batch_xs)
        arr_ys = np.asarray(batch_ys)
        sess.run(train_step, feed_dict={x: arr_xs,
                                        y_: arr_ys})  # this operation should modify the weights and the biases, and also the predicted annotations y
    print("training is over")
    annotationsTargetOldannots = []
    annotationsTargetNewannots = []
    annotationsTargetNewannotsIea = []
    commonGenes = getCommonGenes(annotationsTargetOld, annotationsTargetNew, annotationsTargetNewIea)
    for gene in commonGenes:
        annotationsTargetOldannots.append(getAnnotations(gene, annotationsTargetOld, commonTerms))
        annotationsTargetNewannots.append(getAnnotations(gene, annotationsTargetNew, commonTerms))
        annotationsTargetNewannotsIea.append(getAnnotations(gene, annotationsTargetNewIea, commonTerms))
    print("predicting annotations")
    sigm = sess.run(y, feed_dict={x: np.asarray(annotationsTargetOldannots)})
    print("starting correcting likelihoods,matrix has got " + (str)(len(sigm)) + " genes")
    sigm=correctLikelihoodMatrix(sigm, ancestorList, commonTerms)
    prec=evaluateAccuracy(sigm,annotationsTargetNewannots,annotationsTargetNewannotsIea,tolerance)
    print('prec :', prec)
    writeOnFile(outputfileAccuracy, "a", "prec is: "+str(prec))
    matrix = getArffMatrix(sigm, commonTerms, commonGenes)
    print("writing matrix on file")
    writeMatrixOnFile(outputfileMatrix, matrix)
    return
if __name__ == '__main__':
   main()
