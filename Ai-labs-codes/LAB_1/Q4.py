import pandas as pd

# Load dataset
data = pd.read_csv("students.csv")

print("\n================ STUDENTS WITH MATH MARKS > 50 =================")
# 1. Find how many students have Marks_Math > 50
students_math_above_50 = data[data['Marks_Math'] > 50]
print(students_math_above_50)
print("Number of students with Math marks > 50:", len(students_math_above_50))

print("\n================ STUDENT WITH HIGHEST SCIENCE MARKS =================")
# 2. Find the student with the highest Science marks
highest_science_student = data.loc[data['Marks_Science'].idxmax()]
print(highest_science_student)

print("\n================ CORRELATION BETWEEN MATH AND SCIENCE =================")
# 3. Calculate correlation between Marks_Math and Marks_Science
correlation = data[['Marks_Math', 'Marks_Science']].corr()
print(correlation)
