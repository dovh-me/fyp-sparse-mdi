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
    max_sparsity = 0.5
    min_rtt = 8 # in ms
    max_rtt = min_rtt + 2
    min_pl = 0.01 # in % (0-1)
    rtt_factor = 0.5
    pl_factor = 0 # Assuming no impact from Packet loss
    sparsity_factor = 1
    sparsity_level = 0 # Total Sparsity level for the tensor (only used by sparse encoding strategy)

class AdaptiveSparsityEngine:
    def __init__(self, network_observer: NetworkObservabilityTracker, defaults: SparsityEngineDefaults = SparsityEngineDefaults()):
        """
        Initializes the Adaptive Sparsity Manager.

        :param network_observer: An object responsible for monitoring network conditions.
        :param min_sparsity: Minimum sparsity level.
        :param max_sparsity: Maximum sparsity level.
        """
        self.network_observer = network_observer
        self.defaults = defaults

    def update_defaults(self, defaults: SparsityEngineDefaults):
        self.defaults = defaults
    
    def update_defaults_from_node_config(self, node_config):
        self.defaults.sparsity_factor = node_config.get('sparsity_factor',self.defaults.sparsity_factor)
        self.defaults.max_sparsity = node_config.get('max_sparsity', self.defaults.max_sparsity) 
        self.defaults.min_rtt = node_config.get('min_rtt', self.defaults.min_rtt) 
        self.defaults.max_rtt = node_config.get('max_rtt', self.defaults.max_rtt) 
        self.defaults.min_pl = node_config.get('min_pl', self.defaults.min_pl) 
        self.defaults.rtt_factor = node_config.get('rtt_factor', self.defaults.rtt_factor) 
        self.defaults.pl_factor = node_config.get('pl_factor', self.defaults.pl_factor) 
        self.defaults.sparsity_factor = node_config.get('sparsity_factor', self.defaults.sparsity_factor) 
        self.defaults.sparsity_level = node_config.get('sparsity_level', self.defaults.sparsity_level)

    def compute_tensor_sparsity(self, tensor):
        """
        Calculates the sparsity of a given tensor.

        :param tensor: NumPy array representing the inference activations.
        :return: Sparsity ratio (0 to 1).
        """
        return np.count_nonzero(tensor == 0)
    
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
        non_zero_elements = self.compute_tensor_sparsity(inference_tensor)

        network_factor = self.compute_network_factor()
        added_sparsity_level = min(network_factor, max_sparsity)
        k = non_zero_elements * (1 - added_sparsity_level)
        sparsity_level = 1 - (k / inference_tensor.size) 

        logger.log(f"non_zero_elements:{non_zero_elements} sparsity_level: {sparsity_level} added_sparsity_level: {added_sparsity_level} k: {k}")

        return k, sparsity_level

    def compute_network_factor(self):
        defaults = self.defaults
        rtt_factor = defaults.rtt_factor
        pl_factor = defaults.pl_factor
        sparsity_factor = defaults.sparsity_factor

        # Get Network Metrics
        rtt_norm = self.get_rtt()
        rtt_norm = 1 if rtt_norm <= 0 else rtt_norm

        pl_norm = self.get_pl()
        return sparsity_factor * ((rtt_factor * rtt_norm) + (pl_factor * pl_norm))