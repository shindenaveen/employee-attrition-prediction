import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    df = preprocess_data()
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(r"C:\Users\navee\Downloads\archive\model.pkl", 'wb') as f:
        pickle.dump(model, f)

    return model, X_test, y_test
