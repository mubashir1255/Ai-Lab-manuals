import numpy as np
import matplotlib.pyplot as plt

# Data points
points = {
    "P1": (1,3), "P2": (2,2), "P3": (5,8), "P4": (8,5), "P5": (3,9),
    "P6": (10,7), "P7": (3,3), "P8": (9,4), "P9": (3,7)
}

labels = list(points.keys())
X = np.array(list(points.values()))

# Initial centroids
centroids = [
    np.array(points["P7"]),  # C1
    np.array(points["P9"]),  # C2
    np.array(points["P8"])   # C3
]

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Perform 2 iterations
for _ in range(2):
    clusters = {0: [], 1: [], 2: []}
    for p in X:
        d = [euclidean(p, c) for c in centroids]
        clusters[np.argmin(d)].append(p)

    for i in range(3):
        centroids[i] = np.mean(clusters[i], axis=0)

# Plot
plt.scatter(X[:,0], X[:,1])
for c in centroids:
    plt.scatter(c[0], c[1], marker='x')
for i, txt in enumerate(labels):
    plt.text(X[i,0]+0.05, X[i,1]+0.05, txt)

plt.title("Q1: K-Means From Scratch (K=3)")
plt.show()
