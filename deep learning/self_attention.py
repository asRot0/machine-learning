import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # For numerical stability
    return exp_x / np.sum(exp_x)


def self_attention(input_sequence: np.ndarray) -> np.ndarray:
    """
    Computes self-attention for the given input sequence.

    :param input_sequence: A 2D NumPy array of shape (sequence_length, embedding_dim)
    :return: A 2D NumPy array of the same shape after applying self-attention
    """
    sequence_length, embedding_dim = input_sequence.shape
    output = np.zeros_like(input_sequence)

    for i, pivot_vector in enumerate(input_sequence):
        scores = np.array([np.dot(pivot_vector, vector.T) for vector in input_sequence])
        scores /= np.sqrt(embedding_dim)
        scores = softmax(scores)

        new_pivot_representation = np.sum(input_sequence * scores[:, np.newaxis], axis=0)
        output[i] = new_pivot_representation

    return output


# Example usage
if __name__ == "__main__":
    input_seq = np.array([[1.0, 0.5], [0.2, 0.8], [0.3, 0.7]])
    attention_output = self_attention(input_seq)
    print("Self-Attention Output:\n", attention_output)
