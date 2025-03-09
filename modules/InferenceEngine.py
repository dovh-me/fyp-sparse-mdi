from typing import Callable, Any
import numpy as np

class InferenceEngine:
    def __init__(self, inference_function: Callable[[np.ndarray], np.ndarray]):
        """
        Initializes the inference engine with a custom inference function.

        :param inference_function: A callable that takes an input tensor and returns an output tensor.
        """
        if not callable(inference_function):
            raise ValueError("Inference function must be callable")
        
        self.inference_function = inference_function

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs inference using the provided inference function.

        :param input_tensor: The input tensor.
        :return: The output tensor.
        """
        return self.inference_function(input_tensor)
        
import onnxruntime as ort
import numpy as np

def onnx_inference(input_tensor: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})[0]