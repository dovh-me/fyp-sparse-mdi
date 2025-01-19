import numpy as np

# Function to apply top-k sparsification on activations
def top_k_sparsify(activations, k):
    """
    Apply top-k sparsification to activation vectors.
    
    :param activations: Activation vector (PyTorch tensor).
    :param k: Number of top-k values to retain.
    :return: Sparse activations and their indices.
    """
    activations_np = activations.detach().cpu().numpy()  # Convert to numpy
    original_shape = activations_np.shape
    top_k_indices = np.argpartition(np.abs(activations_np), -k, axis=None)[-k:]  # Get top-k indices
    top_k_values = activations_np.flatten()[top_k_indices]  # Get the top-k values
    total_activations = np.prod(original_shape)
    reconstructed_activations = np.zeros(total_activations)
    reconstructed_activations[top_k_indices] = top_k_values
    
    return reconstructed_activations.view(original_shape) 

# Example usage of top-k sparsification on output_part1
# k = 1000  # Retain top 1000 activations
# sparse_activations, sparse_indices = top_k_sparsify(output_part1, k)
# print("Sparse Activations:", sparse_activations)
# print("Sparse Indices:", sparse_indices)
