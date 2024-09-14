from src.data_processing import load_data, split_data
from src.model_evaluation import evaluate_model
from src.model_training import load_model

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

sgd_clf = load_model('models/sgd_clf_model.pkl')

y_train_pred = sgd_clf.predict(X_train)
precision, recall, f1 = evaluate_model(y_train == 5, y_train_pred)
print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')

