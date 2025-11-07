import numpy as np

c = [1, 2, 3, 4]
print(len(c))
c = np.array(c).reshape(-1,1)
print(np.shape(c))