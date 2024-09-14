import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import load_data, split_data  # Now this will work
from model_training import train_sgd, save_model
from model_training import load_model

# from src.data_processing import load_data, split_data
# from src.model_training import train_sgd, save_model

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

sgd_clf = train_sgd(X_train, y_train == 5)  # Binary classifier for digit 5
save_model(sgd_clf, os.path.join('../models', 'sgd_clf_model.pkl'))
