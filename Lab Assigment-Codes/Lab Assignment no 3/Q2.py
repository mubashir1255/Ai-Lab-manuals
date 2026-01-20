
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'Study Hours': ['Low', 'High', 'High', 'Low', 'High'],
    'Attendance': ['Poor', 'Good', 'Poor', 'Good', 'Good'],
    'Result': ['Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

le = LabelEncoder()

df['Study Hours'] = le.fit_transform(df['Study Hours'])   
df['Attendance'] = le.fit_transform(df['Attendance'])     
df['Result'] = le.fit_transform(df['Result'])             

print("Encoded Data:\n", df, "\n")


X = df[['Study Hours', 'Attendance']]  
y = df['Result']                       

model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)


plt.figure(figsize=(8,6))
plot_tree(model,
          feature_names=['Study Hours', 'Attendance'],
          class_names=['Fail', 'Pass'],
          filled=True,
          rounded=True)
plt.title("Decision Tree - Exam Result Prediction")
plt.show()
