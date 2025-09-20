import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

def load_data(path='../data/iris.csv'):
    return pd.read_csv(path)

def train_model(X_train,y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification Report:\n",classification_report(y_test,y_pred))

def save_model(model,path='../models/logistic_regression.pkl'):
    joblib.dump(model,path)
    print('Model Saved Successfully')

def main():
    df = load_data()
    X = df.drop(columns=['species'])
    y = df['species']

    X_train,X_test,y_train,y_test = train_test_split(
        X,y, test_size=0.2,random_state=42
    )

    model = train_model(X_train=X_train,y_train=y_train)

    evaluate_model(model=model,X_test=X_test,y_test=y_test)

    save_model(model=model)

if __name__=='__main__':
    main()