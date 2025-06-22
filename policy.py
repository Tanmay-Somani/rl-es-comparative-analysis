import numpy as np

class SimpleMLPPolicy:
    def __init__(self, input_dim, output_dim, hidden_dim=8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.params = {
            "W1": np.random.randn(input_dim, hidden_dim) * 0.1,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, output_dim) * 0.1,
            "b2": np.zeros(output_dim)
        }

    def forward(self, x, params=None):
        if params is None:
            params = self.params
        h = np.tanh(np.dot(x, params["W1"]) + params["b1"])
        logits = np.dot(h, params["W2"]) + params["b2"]
        return logits

    def act(self, obs, params=None):
        logits = self.forward(obs, params)
        return int(np.argmax(logits))

    def get_flat(self):
        return np.concatenate([
            self.params["W1"].flatten(),
            self.params["b1"],
            self.params["W2"].flatten(),
            self.params["b2"]
        ])

    def set_flat(self, flat_weights):
        offset = 0
        W1_size = self.input_dim * self.hidden_dim
        b1_size = self.hidden_dim
        W2_size = self.hidden_dim * self.output_dim
        b2_size = self.output_dim

        self.params["W1"] = flat_weights[offset:offset+W1_size].reshape(self.input_dim, self.hidden_dim)
        offset += W1_size
        self.params["b1"] = flat_weights[offset:offset+b1_size]
        offset += b1_size
        self.params["W2"] = flat_weights[offset:offset+W2_size].reshape(self.hidden_dim, self.output_dim)
        offset += W2_size
        self.params["b2"] = flat_weights[offset:offset+b2_size]
