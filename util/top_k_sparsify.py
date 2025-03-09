import numpy as np

# Function to apply top-k sparsification on activations
def top_k_sparsify(activations, k):
    """
    Create a sparse tensor by keeping only the top-k values.
    """
    activations_np = activations.detach().cpu().numpy().flatten()
    k = min(k, activations_np.size)

    top_k_indices = np.argpartition(np.abs(activations_np), -k)[-k:]
    top_k_values = activations_np[top_k_indices]

    # Sort by magnitude for stable results
    sorted_indices = np.argsort(np.abs(top_k_values))
    top_k_values = top_k_values[sorted_indices]
    top_k_indices = top_k_indices[sorted_indices]

    return top_k_values, top_k_indices

# Example usage of top-k sparsification on output_part1
# k = 1000  # Retain top 1000 activations
# sparse_activations, sparse_indices = top_k_sparsify(output_part1, k)
# print("Sparse Activations:", sparse_activations)
# print("Sparse Indices:", sparse_indices)
