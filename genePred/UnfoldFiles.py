import sys
import arff
def removeRepeatingGenesFromAnnotationMatrix(annotationMatrix):
    genes = []
    for gene in annotationMatrix['data']:
        if gene[0] not in genes:
            genes.append(gene[0])
        else:
            annotationMatrix['data'].remove(gene)
def getAnnotationMatrix(path):
    decoder = arff.ArffDecoder()
    f = open(path, 'r')
    matrix = decoder.decode(f, encode_nominal=True, return_type=arff.LOD)
    f.close()
    if ((matrix['data'])[0])[0] == 'DUMMY':
        matrix['data'].pop(0)
    removeRepeatingGenesFromAnnotationMatrix(matrix)
    for i in range(1,len(matrix['attributes'])):
        matrix['attributes'][i]=matrix['attributes'][i][0]
    return matrix
def getAncestorList(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    ancestorList = []
    for row in content:
        anc = row.split()
        ancestorList.append(anc)
    return ancestorList
def unfoldGene(gene, ancestorList, terms):
    for termlist in ancestorList:
        try:
            ind = terms.index(termlist[0])
            if ind in gene:
                for term in termlist:
                    try:
                        newind = terms.index(term)
                        gene[newind] = 1
                    except:
                        None
        except:
            None

def unfoldAnnotationMatrix(annotationMatrix, ancestorList):
    print (len(annotationMatrix['data']))
    for i in range(len(annotationMatrix['data'])):
        if i%100==0:
            print(i)
        unfoldGene(annotationMatrix['data'][i], ancestorList, annotationMatrix['attributes'])
    for i in range(1,len(annotationMatrix['attributes'])):
        annotationMatrix['attributes'][i]=(annotationMatrix['attributes'][i],['0','1'])
def main():

    pathSource = sys.argv[1:][0]
    pathAncestors = sys.argv[1:][1]
    ancestorList = getAncestorList(pathAncestors)
    annotationsSource = getAnnotationMatrix(pathSource)
    unfoldAnnotationMatrix(annotationsSource, ancestorList)
    encoder = arff.ArffEncoder()
    string=encoder.encode(annotationsSource)
    file=open('/home/feroce/Scrivania/gendata/bt_2009_unfolded.arff','w')
    file.write(string)
    file.close()
if __name__ == '__main__':
   main()