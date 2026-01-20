import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("students.csv")

# Bar Chart: Student_ID vs Marks_Math

plt.figure(figsize=(8,5))
plt.bar(data['Student_ID'], data['Marks_Math'], color='skyblue')
plt.xlabel("Student ID")
plt.ylabel("Math Marks")
plt.title("Student ID vs Math Marks")
plt.xticks(data['Student_ID'])  # show all IDs on x-axis
plt.show()

#  Histogram: Age Distribution

plt.figure(figsize=(8,5))
plt.hist(data['Age'], bins=5, color='lightgreen', edgecolor='black')
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.title("Age Distribution of Students")
plt.show()

# Scatter Plot: Marks_Math vs Marks_Science

plt.figure(figsize=(8,5))
plt.scatter(data['Marks_Math'], data['Marks_Science'], color='coral', s=100)
plt.xlabel("Math Marks")
plt.ylabel("Science Marks")
plt.title("Math vs Science Marks")
plt.grid(True)
plt.show()
