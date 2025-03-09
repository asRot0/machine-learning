"""
Convolution Operation Implementation

Formula:
    O(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)

Variables:
    O(i, j): Output feature map at position (i, j)
    I(i+m, j+n): Input image pixels affected by the filter
    K(m, n): Kernel (filter) values applied to the input
"""

import numpy as np


def convolution2d(image, kernel):
    """Performs a 2D convolution operation."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return output


# Example Usage
if __name__ == "__main__":
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example 3x3 image
    kernel = np.array([[1, 0], [0, -1]])  # Example 2x2 kernel
    result = convolution2d(image, kernel)
    print("Convolution Output:\n", result)
