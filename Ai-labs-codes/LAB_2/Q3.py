import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Toy dataset
data = {
    'Hours_Study': [1, 2, 3, 4, 5, 6, 7],
    'Hours_Sleep': [8, 7, 6, 5, 7, 6, 5],
    'Pass': [0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['Hours_Study', 'Hours_Sleep']]
y = df['Pass']

# Fit model
model = LogisticRegression()
model.fit(X, y)

# Predict probability for a new student (fixed)
new_student = pd.DataFrame({'Hours_Study':[30], 'Hours_Sleep':[6]})
prob = model.predict_proba(new_student)[0][1]
print("Probability of Passing:", prob)

# Decision boundary plot
xx, yy = np.meshgrid(np.linspace(0, 35, 300), np.linspace(0, 10, 300))
grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Hours_Study','Hours_Sleep'])
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(df['Hours_Study'], df['Hours_Sleep'], c=df['Pass'], edgecolors='k', cmap=plt.cm.Paired, s=100)
plt.xlabel('Hours Study')
plt.ylabel('Hours Sleep')
plt.title('Logistic Regression Decision Boundary')
plt.grid(True)
plt.show()
