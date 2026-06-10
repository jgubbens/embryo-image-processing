import numpy as np

class OnlineForwardAlgorithm:
    def __init__(self, transition_matrix, n_states):
        self.log_A = np.log(transition_matrix + 1e-9)
        self.n_states = n_states
        self.log_alpha = None

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.log_alpha = np.full(self.n_states, -1e9)
            self.log_alpha[initial_state] = 0.0
        else:
            self.log_alpha = np.log(np.ones(self.n_states) / self.n_states)

    def step(self, emission_probs, temperature):
        log_E = np.log(emission_probs + 1e-9) * temperature
        if self.log_alpha is None:
            self.log_alpha = log_E
        else:
            predicted = self.log_alpha[:, None] + self.log_A
            self.log_alpha = np.logaddexp.reduce(predicted, axis=0) + log_E
        return np.argmax(self.log_alpha)