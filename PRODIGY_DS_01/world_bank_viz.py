import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "dataset/bank_full_cleaned.csv" 
df = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(df.head())

print("\nColumn Information:")
print(df.info())

while True:
    column_name = input("\nEnter the column name you want to visualize {or type 0 to exit}: ")

    if column_name == "0":
        break
    
    elif column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the dataset.")

    else:
        if df[column_name].dtype == 'object':
            plt.figure(figsize=(10, 5))
            sns.countplot(x=column_name, data=df, order=df[column_name].value_counts().index, palette="coolwarm")
            plt.xticks(rotation=45)
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Count')
            plt.show()

        elif pd.api.types.is_numeric_dtype(df[column_name]): 
            plt.figure(figsize=(8, 5))
            sns.histplot(df[column_name], bins=20, kde=True, color='blue')
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.show()

        else:
            print(f"Column '{column_name}' is neither categorical nor numeric.")
