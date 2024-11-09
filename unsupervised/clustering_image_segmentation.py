# clustering_image_segmentation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from skimage.transform import resize

# Load and preprocess image
image = io.imread('https://path_to_your_image.jpg')
image_resized = resize(image, (150, 150))  # Resize for faster processing
X = image_resized.reshape(-1, 3)  # Reshape to RGB values

# Apply K-Means for segmentation
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_image.reshape(image_resized.shape)

# Display the result
plt.imshow(segmented_image)
plt.axis('off')
plt.title('Segmented Image with K-Means')
plt.show()
