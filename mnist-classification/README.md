### Directory Descriptions:

- **data/**: Contains raw datasets like MNIST.
- **notebooks/**: Jupyter notebooks for initial experimentation.
- **scripts/**: Contains scripts for training, evaluating, and predicting models.
- **models/**: Saved models for future use or deployment.
- **src/**: Source code that contains data processing, model training, evaluation, and plotting utilities.
- **plots/**: Stores generated plots and visualizations, such as precision-recall curves.
- **logs/**: Logs for tracking training processes and experiments.
- **tests/**: Unit and integration tests to ensure code correctness.
- **requirements.txt**: Specifies dependencies for the project.
- **README.md**: Project documentation (this file).


```
mnist-classification/
│
├── data/                       # For datasets and raw data
│   └── mnist_784.csv
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   └── exploration.ipynb
│
├── scripts/                    # Main scripts for training and testing models
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/                     # Saved models
│   ├── sgd_clf_model.pkl
│   ├── random_forest_model.pkl
│
├── src/                        # Source files containing model logic
│   ├── __init__.py
│   ├── data_processing.py      # Functions for data preprocessing
│   ├── model_training.py       # Functions for training models
│   ├── model_evaluation.py     # Functions for evaluating models
│   ├── plot_utils.py           # Utility functions for plotting graphs
│   └── config.py               # Configuration variables (paths, params)
│
├── plots/                      # All saved plots will be here
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│
├── logs/                       # Logs for tracking training and experiments
│   └── training_logs.txt
│
├── tests/                      # Unit and integration tests
│   ├── test_data_processing.py
│   ├── test_model_training.py
│
├── .gitignore                  # Files and directories to ignore in Git
├── requirements.txt            # Dependencies required for the project
└── README.md                   # Documentation for the project


░░░░░░░░░░░░░░░░░░░░░░█████████░░░░░░░░░
░░███████░░░░░░░░░░███▒▒▒▒▒▒▒▒███░░░░░░░
░░█▒▒▒▒▒▒█░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒███░░░░
░░░█▒▒▒▒▒▒█░░░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░
░░░░█▒▒▒▒▒█░░░██▒▒▒▒▒██▒▒▒▒▒▒██▒▒▒▒▒███░
░░░░░█▒▒▒█░░░█▒▒▒▒▒▒████▒▒▒▒████▒▒▒▒▒▒██
░░░█████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░░░█▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒██
░██▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒██▒▒▒▒▒▒▒▒▒▒██▒▒▒▒██
██▒▒▒███████████▒▒▒▒▒██▒▒▒▒▒▒▒▒██▒▒▒▒▒██
█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒████████▒▒▒▒▒▒▒██
██▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░
░█▒▒▒███████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██░░░
░██▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█░░░░░
░░████████████░░░█████████████████░░░░░░
```
