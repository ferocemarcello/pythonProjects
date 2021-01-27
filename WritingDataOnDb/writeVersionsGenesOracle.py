import cx_Oracle
import arff
import numpy as np

def getConnection():
    out=1
    conn_str = u'bio/password@geneserver.cpm6yh9ngaso.us-west-2.rds.amazonaws.com:1521/GENEBASE'
    while out==1:
        try:
            conn = cx_Oracle.connect(conn_str)
            out = 0
        except Exception as e:
            print e
    return conn

def writeVersionOracle(id, year, iea, speciesName):
    conn=getConnection()
    cur = conn.cursor()
    try:
        cur.execute("""INSERT INTO VERSIONS VALUES (:1, :2,:3,:4)""", {"1": id,"2": year,"3": iea,"4": speciesName,})
        conn.commit()
    except Exception as e:
        print e
    cur.close()
    conn.close()

def writeGeneOracle( id,geneName,idversion):
    conn = getConnection()
    cur = conn.cursor()
    try:
        cur.execute("""INSERT INTO GENES VALUES (:1,:2,:3)""", {"1": id,"2": geneName,"3": idversion})
        conn.commit()
    except Exception as e:
        print e
    cur.close()
    conn.close()
def getnewAnnotations(terms,annotations,idgene):
    conn = getConnection()
    cur = conn.cursor()
    rows = []
    for termindex in range(0, len(terms)):
        term = (terms[termindex])[0]
        value = annotations[termindex]
        try:
            rows.append([idgene, term, value])
        except Exception as e:
            print e
    return rows
def writeAnnotationsOracle(terms,annotations,idgene):
    rows=[]
    for termindex in range(0, len(terms)):
        term = (terms[termindex])[0]
        value = annotations[termindex]
        try:
            rows.append([idgene,term,value])
        except Exception as e:
            print e
def writeAnnotationsFromList(annotationList):
    conn = getConnection()
    cur = conn.cursor()
    try:
        cur.executemany("""INSERT INTO ANNOTATIONS VALUES(:1,:2,:3)""", annotationList)
        conn.commit()
    except Exception as e:
        print e
    cur.close()
    conn.close()
def writeTermOracle( termName):
    conn = getConnection()
    cur = conn.cursor()
    try:
        cur.execute("""INSERT INTO TERMS VALUES (:1, null)""", {"1": termName})
    except Exception as e:
        print e
    conn.commit()
    cur.close()
    conn.close()

def writeTermsOracle(terms):
    conn = getConnection()
    cur = conn.cursor()
    rows=[]
    try:
        cur.execute("""select NAME from TERMS""")
        existingterms = cur.fetchall()
    except Exception as e:
        print e

    existingterms=np.array(existingterms)
    for term in terms:
        if term[0] not in existingterms:
            rows.append([term[0], 'null'])

    try:
        cur.executemany("""INSERT INTO TERMS VALUES(:1,:2)""", rows)
        conn.commit()
    except Exception as e:
        print e
    cur.close()
    conn.close()

def writeGenesOracle(genes,idversion):
    rows=[]
    rows.append([-1, 'dummy', -1])
    genenamearray=np.array(genes)[:,0]
    genenamelist=list(genenamearray)
    newid=getMaxGeneId()+1
    for geneindex in range(0, len(genes)):
        genevers=[genes[geneindex][0],idversion]
        if list(np.array(rows)[:,1]).count(genevers[0])<1:
            rows.append([newid, genevers[0], idversion])
            newid += 1
    rows.pop(0)
    conn = getConnection()
    cur = conn.cursor()
    try:
        cur.executemany("""INSERT INTO GENES VALUES (:1,:2,:3)""", rows)
        conn.commit()
    except Exception as e:
        print e
    cur.close()
    conn.close()

def getMaxGeneId():
    try:
        conn=getConnection()
        cur=conn.cursor()
        cur.execute("""select MAX (ID) from GENES""")
        maxid = cur.fetchone()[0]
        if maxid==None:
            maxid=0
    except Exception as e:
        print e
    return maxid

def getGeneId(genename,versionnumber):
    try:
        conn=getConnection()
        cur=conn.cursor()
        geneid=0
        cur.execute("""select ID from GENES where name=:1 and ORGANISMVERSION=:2""",{"1": genename,"2":versionnumber})
        geneid = cur.fetchone()[0]
    except Exception as e:
        print e
    return geneid

def writeGenesVersionsOracleFile(idversion,year,iea,species,path):
    writeVersionOracle(idversion, year, iea, species)
    arfffile=arff.load(open(path,'rb'))
    genes=arfffile['data']
    genes.pop(0)
    terms = arfffile['attributes']
    terms.pop(0)
    writeTermsOracle(terms)
    rows=[]
    idgene=0
    redudantgenes=[]
    #onlygenenames=list(np.array(genes)[:,0])
    onlygenenames=[]
    for gene in genes:
        onlygenenames.append(gene[0])

    writeGenesOracle(genes,idversion)
    startingindex=0
    for gene in genes:
        name = gene[0]

        if onlygenenames.count(name)==1:
            gene.pop(0)
            if idgene==0:
                idgene = getGeneId(name, idversion)
                startingindex=idgene
            if (idgene-startingindex)%100==0:
                print (idgene-startingindex)
            rows.extend(getnewAnnotations(terms, gene, idgene))
            if len(rows) >= 1000000:
                writeAnnotationsFromList(rows)
                rows=[]
            idgene += 1

        if onlygenenames.count(name)>1 and redudantgenes.count(name)<1:
            redudantgenes.append(name)
            gene.pop(0)
            if idgene == 0:
                idgene = getGeneId(name, idversion)
                startingindex = idgene
            if (idgene-startingindex)%100==0:
                print (idgene-startingindex)
            rows.extend(getnewAnnotations(terms, gene, idgene))
            if len(rows)>=1000000:
                writeAnnotationsFromList(rows)
                rows = []
            idgene += 1

        #writeAnnotationsOracle(terms,gene,idgene)
    writeAnnotationsFromList(rows)

def checkgeneLength(path1,path2):
    arff1 = arff.load(open(path1, 'rb'))
    genes1 = arff1['data']
    genes1.pop(0)
    onlygenenames1 = []

    for gene in genes1:
        onlygenenames1.append(gene[0])

    arff2 = arff.load(open(path2, 'rb'))
    genes2 = arff2['data']
    genes2.pop(0)
    onlygenenames2 = []

    for gene in genes2:
        onlygenenames2.append(gene[0])

    for gene in genes1:
        name = gene[0]
        if onlygenenames1.count(name) >1:
                onlygenenames1.remove(name)

    for gene in genes2:
        name = gene[0]
        if onlygenenames2.count(name) >1:
                onlygenenames2.remove(name)
    print len(onlygenenames1)
    print len(onlygenenames2)
#checkgeneLength('/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2009.arff','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2013.arff')
#checkgeneLength('/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2009.arff','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2013.arff')
#writeGenesVersionsOracleFile(1, 2009, 0, 'Bos Taurus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2009.arff')
#writeGenesVersionsOracleFile(2, 2013, 0, 'Bos Taurus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2013.arff')
#writeGenesVersionsOracleFile(3, 2009, 1, 'Bos Taurus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2009_IEA.arff')
writeGenesVersionsOracleFile(4, 2013, 1, 'Bos Taurus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/bt_2013_IEA.arff')

#writeGenesVersionsOracleFile(5, 2009, 0, 'Mus Musculus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2009.arff')
#writeGenesVersionsOracleFile(6, 2013, 0, 'Mus Musculus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2013.arff')
#writeGenesVersionsOracleFile(7, 2009, 1, 'Mus Musculus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2009_IEA.arff')
#writeGenesVersionsOracleFile(8, 2013, 1, 'Mus Musculus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/mm_2013_IEA.arff')

#writeGenesVersionsOracleFile(9, 2009, 0, 'Rattus Norvegicus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/rn_2009.arff')
#writeGenesVersionsOracleFile(10, 2013, 0, 'Rattus Norvegicus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/rn_2013.arff')
#writeGenesVersionsOracleFile(11, 2009, 1, 'Rattus Norvegicus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/rn_2009_IEA.arff')
#writeGenesVersionsOracleFile(12, 2013, 1, 'Rattus Norvegicus','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/rn_2013_IEA.arff')

#writeGenesVersionsOracleFile(13, 2009, 0, 'Homo Sapiens','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/hs_2009.arff')
#writeGenesVersionsOracleFile(14, 2013, 0, 'Homo Sapiens','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/hs_2013.arff')
#writeGenesVersionsOracleFile(15, 2013, 1, 'Homo Sapiens','/media/marcelloferoce/DATI/marcello/universita/tesi/gendata/hs_2013_IEA.arff')