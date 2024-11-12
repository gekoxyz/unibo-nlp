import numpy as np

features = np.zeros((10,10))

features[5, :2] = np.array([7, 7])

print(features)