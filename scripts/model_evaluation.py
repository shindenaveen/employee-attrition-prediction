import pickle
from sklearn.metrics import classification_report

def evaluate_model():
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    with open(r"C:\Users\navee\Downloads\archive\classification_report.txt", 'w') as f:
        f.write(report)

    print(report)

evaluate_model()
