# fl_rl_agent.py
import numpy as np
import tensorflow as tf
from rl_agent import DQNAgent # Re-use the DQN agent for clients

class FLClient:
    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
    
    def get_weights(self):
        return self.agent.model.get_weights()
        
    def set_weights(self, weights):
        self.agent.model.set_weights(weights)
        self.agent.update_target_model()

class FLAgent:
    def __init__(self, state_size, action_size, num_clients=5):
        self.num_clients = num_clients
        self.state_size = state_size
        self.action_size = action_size
        
        # The "server" model
        self.global_model = DQNAgent(state_size, action_size).model
        
        # Create clients
        self.clients = [FLClient(state_size, action_size) for _ in range(num_clients)]
        self.sync_clients()

    def sync_clients(self):
        """Send global model weights to all clients."""
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)
            
    def aggregate_weights(self, client_weights_list):
        """Federated Averaging."""
        # Calculate the average of the weights
        avg_weights = list()
        for weights_list_tuple in zip(*client_weights_list):
            avg_weights.append(np.array([np.array(w) for w in weights_list_tuple]).mean(axis=0))
        
        return avg_weights

    def communication_cost(self, weights):
        """Calculate the size of the model weights in bytes."""
        total_bytes = 0
        for layer in weights:
            total_bytes += layer.nbytes
        return total_bytes

    def global_round(self):
        """
        One round of federated learning.
        1. Sync clients with the global model.
        2. Clients train locally (this logic is handled in the runner).
        3. Aggregate client model updates.
        """
        client_weights = [client.get_weights() for client in self.clients]
        
        # Aggregate weights from all clients
        aggregated_weights = self.aggregate_weights(client_weights)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        # Send updated global model back to clients
        self.sync_clients()

        # Calculate communication cost (size of one client's upload)
        cost = self.communication_cost(client_weights[0])
        return cost