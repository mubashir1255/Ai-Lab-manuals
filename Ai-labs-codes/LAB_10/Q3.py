import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    (1,3),(2,2),(5,8),(8,5),(3,9),
    (10,7),(3,3),(9,4),(3,7),(6,2)
])

labels = ["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10"]

model = KMeans(n_clusters=3, random_state=42, n_init=10)
y = model.fit_predict(X)

plt.scatter(X[:,0], X[:,1])
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='x')

for i, txt in enumerate(labels):
    plt.text(X[i,0]+0.05, X[i,1]+0.05, txt)

plt.title("Q3: K=3 with New Point P10")
plt.show()

print("P10 belongs to cluster:", y[-1])
print("New centroids:\n", model.cluster_centers_)
