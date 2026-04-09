import pandas as pd
import math

# Dataset
data = pd.DataFrame({
    'A1': ['T','T','T','F','F','F'],
    'A2': ['T','T','F','F','T','T'],
    'Class': ['+','+','-','+','-','-']
})

def entropy(col):
    values = col.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in values)

def info_gain(data, attr, target='Class'):
    total_entropy = entropy(data[target])

    values = data[attr].unique()
    weighted_entropy = 0

    for v in values:
        subset = data[data[attr] == v]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# Calculate IG
print("Information Gain:")
for col in ['A1','A2']:
    print(col, ":", round(info_gain(data, col), 3))

# Simple tree representation
def build_tree(data, attrs):
    if len(data['Class'].unique()) == 1:
        return data['Class'].iloc[0]

    if not attrs:
        return data['Class'].mode()[0]

    gains = {attr: info_gain(data, attr) for attr in attrs}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    for val in data[best_attr].unique():
        subset = data[data[best_attr] == val]
        subtree = build_tree(subset, [a for a in attrs if a != best_attr])
        tree[best_attr][val] = subtree

    return tree

tree = build_tree(data, ['A1','A2'])
print("\nDecision Tree:")
print(tree)
