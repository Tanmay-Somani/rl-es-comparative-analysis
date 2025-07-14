# es_agent.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class ESModel:
    """
    The neural network for an individual member of the population.
    This class does not need any changes.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='softmax'))
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

class ESAgent:
    """
    The main agent that manages the population and evolution.
    This class has been corrected.
    """
    def __init__(self, state_size, action_size, population_size=50, sigma=0.1, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.sigma = sigma # Noise standard deviation
        self.learning_rate = learning_rate
        
        self.master_model = ESModel(state_size, action_size)
        self.population = [ESModel(state_size, action_size) for _ in range(population_size)]
        
        # This will store the random noise applied to each member of the population.
        # This is crucial for the corrected evolution step.
        self.perturbations = []

    def evolve(self, rewards):
        """
        Updates the master model's weights based on the performance of the population.
        This is the corrected evolution logic.
        """
        # 1. Normalize rewards. This stabilizes the update step by turning scores
        #    into a standardized distribution. High scores get positive values, low scores get negative.
        rewards = np.array(rewards)
        # Add a small value (epsilon) to the standard deviation to prevent division by zero
        # if all rewards in a generation are identical.
        if np.std(rewards) > 1e-7:
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        else:
            # No information to learn from if all rewards are the same, so we can skip the update.
            return

        # 2. Initialize a placeholder for the total, aggregated update.
        #    It has the same shape as the model's weights, but is filled with zeros.
        master_weights = self.master_model.get_weights()
        total_update = [np.zeros_like(w) for w in master_weights]

        # 3. Calculate the "gradient" by scaling each individual's perturbation
        #    by its normalized reward and summing them all up.
        for index, p_noise in enumerate(self.perturbations):
            for i in range(len(total_update)):
                total_update[i] += p_noise[i] * rewards[index]
        
        # 4. Apply the single, aggregated update to the master model.
        #    This is the "learning" step.
        update_factor = self.learning_rate / (self.population_size * self.sigma)
        for i in range(len(master_weights)):
            master_weights[i] += update_factor * total_update[i]

        self.master_model.set_weights(master_weights)

    def generate_population(self):
        """
        Creates a new population by adding random noise to the master model's weights.
        This now also stores the noise for use in the evolve step.
        """
        master_weights = self.master_model.get_weights()
        self.perturbations = [] # Clear perturbations from the previous generation

        for p in self.population:
            new_weights = []
            p_noise = [] # The perturbation for this specific individual

            for layer_weights in master_weights:
                # Generate random noise from a normal distribution
                noise = np.random.randn(*layer_weights.shape) * self.sigma
                p_noise.append(noise)
                
                # Apply the noise to the master weights to create the new individual's weights
                new_weights.append(layer_weights + noise)
            
            p.set_weights(new_weights)
            self.perturbations.append(p_noise) # Store the noise that was used

        return self.population