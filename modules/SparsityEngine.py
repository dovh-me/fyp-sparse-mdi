import numpy as np
import os
import sys

module_path = os.path.abspath('../')
sys.path.insert(0, module_path)

from modules.NetworkObservabilityTracker import NetworkObservabilityTracker
from util.logger import logger
from dataclasses import dataclass

@dataclass
class SparsityEngineDefaults:
    min_sparsity = 0.2
    max_sparsity = 0.7
    min_rtt = 8 # in ms
    max_rtt = min_rtt + 2
    min_pl = 0.01 # in % (0-1)
    rtt_factor = 1
    pl_factor = 0 # Assuming no impact from Packet loss

class SparsityEngine:
    def __init__(self, network_observer: NetworkObservabilityTracker, defaults: SparsityEngineDefaults = SparsityEngineDefaults()):
        """
        Initializes the Adaptive Sparsity Manager.

        :param network_observer: An object responsible for monitoring network conditions.
        :param min_sparsity: Minimum sparsity level.
        :param max_sparsity: Maximum sparsity level.
        """
        self.network_observer = network_observer
        self.defaults = defaults

    def compute_tensor_sparsity(self, tensor):
        """
        Calculates the sparsity of a given tensor.

        :param tensor: NumPy array representing the inference activations.
        :return: Sparsity ratio (0 to 1).
        """
        non_zero_elements = np.count_nonzero(tensor == 0)
        total_elements = tensor.size
        return non_zero_elements, 1 - (non_zero_elements / total_elements)
    
    def get_rtt(self):
        network_metrics = self.network_observer.get_rtt()
        defaults = self.defaults

        max_rtt = network_metrics.get("max", defaults.max_rtt)  # Default 50ms
        min_rtt = network_metrics.get("min", defaults.min_rtt)  # Default 50ms
        rtt = network_metrics.get("current", defaults.min_rtt + 1)  # Default 50ms
        value = 1 - ((rtt - min_rtt) / (max_rtt - min_rtt))
        
        print(f"1 - ((rtt - min_rtt) / (max_rtt - min_rtt)): 1 - (({rtt} - {min_rtt}) / ({max_rtt} - {min_rtt})) = {value}")
        # Normalize the rtt
        return  value

    def get_pl(self):
        return 0

    def compute_sparsity_level(self, inference_tensor):
        """
        Determines the optimal sparsity level based on network status and inference tensor.

        :param inference_tensor: NumPy array representing activations.
        :return: Recommended sparsity level (0 to 1).
        """
        defaults = self.defaults
        max_sparsity = defaults.max_sparsity

        # Compute Tensor Sparsity
        non_zero_elements, sparsity_level = self.compute_tensor_sparsity(inference_tensor)

        network_factor = self.compute_network_factor()
        k = non_zero_elements * network_factor 
        min_k = non_zero_elements * max_sparsity

        k = max(min_k, k)

        logger.log(f"rtt: {self.network_observer.rtt} sparsity_level: {sparsity_level} k: {k}")

        return k, sparsity_level

    def compute_network_factor(self):
        defaults = self.defaults
        rtt_factor = defaults.rtt_factor
        pl_factor = defaults.pl_factor

        # Get Network Metrics
        rtt_norm = self.get_rtt()
        rtt_norm = 1 if rtt_norm <= 0 else rtt_norm

        pl_norm = self.get_pl()
        return ((rtt_factor * rtt_norm) + (pl_factor * pl_norm))