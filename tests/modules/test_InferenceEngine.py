import pytest
import numpy as np
import os
import sys

module_path = os.path.abspath('./')
sys.path.insert(0, module_path)

from modules.InferenceEngine import InferenceEngine

def mock_inference_function(input_tensor: np.ndarray) -> np.ndarray:
    """Mock inference function that doubles the input tensor values."""
    return input_tensor * 2

@pytest.fixture
def inference_engine():
    """Fixture to create an InferenceEngine instance with a mock function."""
    return InferenceEngine(mock_inference_function)

def test_inference_engine_initialization():
    """Test that the InferenceEngine initializes correctly with a callable function."""
    engine = InferenceEngine(mock_inference_function)
    assert engine.inference_function is not None

def test_inference_engine_invalid_function():
    """Ensure an error is raised if a non-callable is provided."""
    with pytest.raises(ValueError):
        InferenceEngine("not_a_function")

def test_infer(inference_engine):
    """Test that the inference function correctly transforms the input tensor."""
    input_tensor = np.array([1, 2, 3])
    output_tensor = inference_engine.infer(input_tensor)
    expected_output = np.array([2, 4, 6])

    assert np.array_equal(output_tensor, expected_output)

def test_empty_input(inference_engine):
    """Ensure the inference function handles empty tensors gracefully."""
    input_tensor = np.array([])
    output_tensor = inference_engine.infer(input_tensor)
    
    assert output_tensor.size == 0  # Should return an empty tensor

@pytest.mark.parametrize("input_tensor,expected_output", [
    (np.array([[1, 2], [3, 4]]), np.array([[2, 4], [6, 8]])),
    (np.array([0, 0, 0]), np.array([0, 0, 0])),
    (np.array([-1, -2, -3]), np.array([-2, -4, -6])),
])
def test_infer_various_inputs(inference_engine, input_tensor, expected_output):
    """Test inference with various input tensors."""
    output_tensor = inference_engine.infer(input_tensor)
    assert np.array_equal(output_tensor, expected_output)

def test_onnx_inference(monkeypatch):
    """Mock ONNX inference to test its integration."""
    import onnxruntime as ort
    
    class MockSession:
        """Mock ONNX session that simulates inference."""
        def __init__(self, model_path, providers):
            pass
        
        def get_inputs(self):
            class MockInput:
                name = "input"
            return [MockInput()]
        
        def run(self, output_names, inputs):
            return [inputs["input"] * 2]  # Simulating an ONNX model that doubles inputs

    monkeypatch.setattr(ort, "InferenceSession", MockSession)

    from modules.InferenceEngine import onnx_inference
    input_tensor = np.array([1, 2, 3])
    output_tensor = onnx_inference(input_tensor)
    expected_output = np.array([2, 4, 6])

    assert np.array_equal(output_tensor, expected_output)