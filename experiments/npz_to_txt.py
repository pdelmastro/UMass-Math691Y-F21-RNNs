import numpy as np
import sys
data = np.load("Lorenz.npz")
print(data.files)
row = data.files
print(data["y"])
np.set_printoptions(threshold=np.inf)
sys.stdout=open("Lorenz.txt","w")
for i in row:
    print(data[i])
sys.stdout.close()