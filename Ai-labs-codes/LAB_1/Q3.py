import pandas as pd

# Load the dataset
data = pd.read_csv("students.csv")

print("\n================ DATASET STRUCTURE =================")
# 1. Dataset structure
data.info()

print("\n================ SUMMARY STATISTICS =================")
# 2. Summary statistics
print(data.describe())

print("\n================ MEAN OF MATH MARKS =================")
# 3. Mean of Math marks
mean_math = data['Marks_Math'].mean()
print("Mean of Math Marks:", mean_math)

print("\n================ MAXIMUM SCIENCE MARKS =================")
# 4. Maximum Science marks
max_science = data['Marks_Science'].max()
print("Maximum Science Marks:", max_science)
