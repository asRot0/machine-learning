from data_processing import DataProcessor
from cluster_model import ClusteringModel
from pca_processor import PCAProcessor


if __name__ == "__main__":
    # Data Preparation
    data_path = "../data/seeds/seeds_dataset.txt"
    cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
    processor = DataProcessor(data_path, cols)
    print(processor.preview_data())

    # Clustering Model
    X, _ = processor.get_features_and_labels(["compactness", "asymmetry"], "class")
    clustering_model = ClusteringModel(n_clusters=3)
    clustering_model.fit(X)
    clustering_model.visualize_clusters(X, "compactness", "asymmetry")

    # PCA
    X_full, y_full = processor.get_features_and_labels(cols[:-1], "class")
    pca_processor = PCAProcessor(n_components=2)
    X_pca = pca_processor.fit_transform(X_full)
    pca_processor.visualize_pca(X_pca, y_full)
