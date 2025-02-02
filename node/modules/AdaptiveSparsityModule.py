from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
import numpy as np

class AdaptiveSparsityManager:
    def __init__(self, network_observer, min_sparsity=0.2, max_sparsity=0.8):
        """
        Initializes the Adaptive Sparsity Manager.

        :param network_observer: An object responsible for monitoring network conditions.
        :param min_sparsity: Minimum sparsity level.
        :param max_sparsity: Maximum sparsity level.
        """
        self.network_observer = network_observer
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity

        # OpenTelemetry Metrics Setup
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(__name__)
        self.sparsity_metric = self.meter.create_up_down_counter(
            name="adaptive_sparsity_level",
            description="Dynamically adjusted sparsity level based on network and inference metrics",
        )

        # OTLP Exporter
        self.exporter = OTLPMetricExporter(endpoint="http://localhost:4317")
        self.meter.register_metric_reader(self.exporter)

    def compute_tensor_sparsity(self, tensor):
        """
        Calculates the sparsity of a given tensor.

        :param tensor: NumPy array representing the inference activations.
        :return: Sparsity ratio (0 to 1).
        """
        zero_elements = np.count_nonzero(tensor == 0)
        total_elements = tensor.size
        return zero_elements / total_elements if total_elements > 0 else 0

    def compute_sparsity_level(self, inference_tensor):
        """
        Determines the optimal sparsity level based on network status and inference tensor.

        :param inference_tensor: NumPy array representing activations.
        :return: Recommended sparsity level (0 to 1).
        """
        # Get Network Metrics
        network_metrics = self.network_observer.get_network_status()
        rtt = network_metrics.get("rtt", 50)  # Default 50ms
        packet_loss = network_metrics.get("packet_loss", 0.01)  # Default 1%

        # Compute Tensor Sparsity
        tensor_sparsity = self.compute_tensor_sparsity(inference_tensor)

        # Compute Adaptive Sparsity Level
        network_penalty = min(1.0, (rtt / 100.0) + (packet_loss * 10))  # Normalize impact
        adaptive_sparsity = self.min_sparsity + (self.max_sparsity - self.min_sparsity) * (1 - network_penalty)
        adaptive_sparsity = max(self.min_sparsity, min(self.max_sparsity, adaptive_sparsity + tensor_sparsity))

        # Export to OpenTelemetry
        self.sparsity_metric.add(adaptive_sparsity, {})

        return adaptive_sparsity