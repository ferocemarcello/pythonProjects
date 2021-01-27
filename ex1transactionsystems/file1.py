#Marcello Feroce, enrolment number: 03707034, TUM ID: ge68mij
#this script should be called in this way: python3 file1.py w1x,r2x,w2y,r3y,w3z,r1z
import sys#this import is necessary to take inputs from command line

#here down with these lines, I handle the input from command line, and I assign each operation to a different element of the array H
arr = sys.argv[1].split(',')
H=[]
H.append(arr[0])
H.append(arr[1])
H.append(arr[2])
H.append(arr[3])
H.append(arr[4])
H.append(arr[5])

serializable=True
for i in range(len(H)):
    for j in range(len(H)):
        if i!=j and (H[i])[2]==(H[j])[2] and (H[i])[1]!=(H[j][1]) and ((H[i])[0]=="w" or (H[j])[0]):#with this I check if two different operations work on the same resource and if they belong to different transactions and if one of them is a write
            serializable = False#if the previous statement is true, it means the history is not serializable
            break
    if serializable == False:
        break
print(serializable)