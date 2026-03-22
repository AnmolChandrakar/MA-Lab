import numpy as np

X = np.array([[1,2],[1,3],[1,4]])
Y = np.array([[1],[2],[3]])

I = np.array([[0,0],[0,1]])
B = np.dot(np.linalg.inv((np.dot(X.T, X)) + I), np.dot(X.T, Y))

print("coeff. vector B = ",B)
print(f"line of regression : y = {B[0][0]:.2f} + {B[1][0]:.2f}x")