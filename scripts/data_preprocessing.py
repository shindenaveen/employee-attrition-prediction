import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    filepath = r"C:\Users\navee\Downloads\archive\HR-Employee-Attrition.csv"
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df
