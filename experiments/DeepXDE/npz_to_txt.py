import numpy as np
import sys
data = np.load("Lorenz.npz")
row = data.files
print(row)
print(data)
np.set_printoptions(threshold=np.inf)
sys.stdout=open("Lorenz.txt","w")
print(row)
for i in row:
    print(data[i])
sys.stdout.close()