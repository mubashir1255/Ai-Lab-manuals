import math
import pandas as pd


data = {
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'Study Hours': ['Low', 'High', 'High', 'Low', 'High'],
    'Attendance': ['Poor', 'Good', 'Poor', 'Good', 'Good'],
    'Result': ['Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")


def entropy(column):
    values = column.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in values)


target_entropy = entropy(df['Result'])
print(f"1. Entropy(Result) = {target_entropy:.3f}")

def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = 0
    
    for v in values:
        subset = df[df[attribute] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    
    info_gain = total_entropy - weighted_entropy
    return info_gain

ig_study = information_gain(df, 'Study Hours', 'Result')
ig_attendance = information_gain(df, 'Attendance', 'Result')

print(f"2. Information Gain (Study Hours) = {ig_study:.3f}")
print(f"   Information Gain (Attendance) = {ig_attendance:.3f}")


best_attr = 'Study Hours' if ig_study > ig_attendance else 'Attendance'
print(f"3. Best attribute for root node = {best_attr}")
