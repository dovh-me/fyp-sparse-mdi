class NetworkObservabilityTracker :
    def __init__(self):
        self.inference_ingress = 0
        self.inference_egress = 0

        self.rtt_with_next_node = 0
        self.rtt = {
            'min': 0,
            'max': 0,
            'current': 0
        }

        self.packet_loss = 0
        
        self.inference_counter = 0
        self.sparsity_counter = 0

    def get_inference_metrics(self):
        return {
            # 'values': self.values_bytes_transferred,
            # 'indices': self.indices_bytes_transferred,
            'ingress': self.inference_ingress,
            'egress': self.inference_egress,
            'rtt': self.rtt['current']
        }
    
    def update_rtt_with_next_node(self, rtt: int):
        self.rtt['max'] = rtt if self.rtt['max'] < rtt else self.rtt['max']
        self.rtt['min'] = rtt if self.rtt['min'] > rtt else self.rtt['min']
        self.rtt['current'] = rtt
        self.rtt_with_next_node = rtt

    def get_rtt(self):
        return self.rtt

    def update_ingress_metrics(self, bytes):
        self.inference_ingress += len(bytes)

    def update_egress_metrics(self, bytes):
        self.inference_egress += len(bytes)

    def update_inference_counter(self):
        self.inference_counter += 1
    
    def update_sparsity_counter(self):
        self.sparsity_counter += 1

    def update_packet_loss(self, loss: int):
        self.packet_loss = loss