# Helper function to convert PyTorch tensors to NumPy arrays
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if hasattr(tensor, 'requires_grad') else tensor.cpu().numpy()