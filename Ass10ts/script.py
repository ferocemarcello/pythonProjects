import random
import copy

def mod(d,i):
    if d==0:
        return d,i
    return 0,1
def dele():
    return 0
def ins():
    return 1
a=0
b=0
c=0
a2=0
b2=0
c2=0
arr=[0,0,0,0,0,0,0]
arr2=[0,0,0,0,0,0,0]
done=False
while(not done):
    for op in range(len(arr)):
        arr[op]=(random.sample(["del","ins","mod"],1),(random.sample(['a','b','c'],2)))
    arr2=copy.deepcopy(arr)
    for op in range(len(arr2)):
        if (arr2[op])[0] == ["mod"]:
            deleted=((arr2[op])[1])[0]
            inserted = ((arr2[op])[1])[1]
            (arr2[op])=(["del"],[deleted,inserted])
            arr2.insert(op+1,(["ins"],[inserted,deleted]))
    for op in range(len(arr2)):
        if (arr2[op])[0]==["ins"]:
            if ((arr2[op])[1])[0]=='a':
                a2=ins()
            if ((arr2[op])[1])[0]=='b':
                b2=ins()
            if ((arr2[op])[1])[0]=='c':
                c2=ins()
        if (arr2[op])[0]==["del"]:
            if ((arr2[op])[1])[0]=='a':
                a2=dele()
            if ((arr2[op])[1])[0]=='b':
                b2=dele()
            if ((arr2[op])[1])[0]=='c':
                c2=dele()
    for op in range(len(arr)):
        if (arr[op])[0]==["ins"]:
            if ((arr[op])[1])[0]=='a':
                a=ins()
            if ((arr[op])[1])[0]=='b':
                b=ins()
            if ((arr[op])[1])[0]=='c':
                c=ins()
        if (arr[op])[0]==["del"]:
            if ((arr[op])[1])[0]=='a':
                a=dele()
            if ((arr[op])[1])[0]=='b':
                b=dele()
            if ((arr[op])[1])[0]=='c':
                c=dele()
        if (arr[op])[0]==["mod"]:
            if ((arr[op])[1])[0]=='a' and ((arr[op])[1])[1]=='b':
                a,b=mod(a,b)
            if ((arr[op])[1])[0]=='a' and ((arr[op])[1])[1]=='c':
                a,c=mod(a,c)
            if ((arr[op])[1])[0]=='c' and ((arr[op])[1])[1]=='b':
                c,b=mod(c,b)
            if ((arr[op])[1])[0]=='c' and ((arr[op])[1])[1]=='a':
                c,a=mod(c,a)
            if ((arr[op])[1])[0]=='b' and ((arr[op])[1])[1]=='a':
                b,a=mod(b,a)
            if ((arr[op])[1])[0]=='b' and ((arr[op])[1])[1]=='c':
                b,c=mod(b,c)
    if a!=a2 and b!=b2 and c!=c2:
        print(a, b, c)
        print(a2, b2, c2)
        print(arr)
        print(arr2)
        done=True