import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = {
    "P1": (1,3), "P2": (2,2), "P3": (5,8), "P4": (8,5), "P5": (3,9),
    "P6": (10,7), "P7": (3,3), "P8": (9,4), "P9": (3,7)
}

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

C1 = np.array((3,3))
C2 = np.array((3,7))
C3 = np.array((9,4))

rows = []
for name, p in points.items():
    p = np.array(p)
    d1 = euclidean(p, C1)
    d2 = euclidean(p, C2)
    d3 = euclidean(p, C3)
    cluster = "C" + str(np.argmin([d1, d2, d3]) + 1)
    rows.append([name, d1, d2, d3, cluster])

df = pd.DataFrame(rows, columns=[
    "Point", "Dist to C1", "Dist to C2", "Dist to C3", "Assigned Cluster"
])

print(df)

# Plot first iteration
X = np.array(list(points.values()))
plt.scatter(X[:,0], X[:,1])
plt.scatter([3,3,9], [3,7,4], marker='x')
for i, txt in enumerate(points.keys()):
    plt.text(X[i,0]+0.05, X[i,1]+0.05, txt)

plt.title("Q4: First Iteration Plot")
plt.show()
