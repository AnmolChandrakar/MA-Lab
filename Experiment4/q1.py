import numpy as np
from sklearn import svm

# Define the dataset with the corrected point
X = np.array([
    [1, 1],
    [2, 1],  # Corrected from (2,3)
    [2, 3],
    [3, 3]
])
y = np.array([1, 1, -1, -1])

# Create a linear SVM classifier
# We set C to a large value to ensure a hard margin if possible, for demonstration
clf = svm.SVC(kernel='linear', C=1000)

# Train the SVM model
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
b = clf.intercept_[0]

# Predict labels for the training data itself
predicted_labels = clf.predict(X)

print(f"Vector w: {w}")
print(f"Bias b: {b}")
print(f"Predicted Labels: {predicted_labels}")