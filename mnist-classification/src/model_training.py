from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import joblib

def train_sgd(X_train, y_train):
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
