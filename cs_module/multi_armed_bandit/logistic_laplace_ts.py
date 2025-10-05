import numpy as np


# Logistic Thompson Sampling with Laplace approximation for Bayesian logistic regression.
# Based on Chapelle & Li (NIPS 2011), Algorithm 3.

# Reward is converted to binary (0 or 1).


class LogisticLaplaceTS:
    def __init__(self, feature_dim, discount=1):
        self.feature_dim = feature_dim

        # Prior: θ ~ N(0, I)  (i.e., Var = 1 on each coordinate)
        self.mu = np.zeros(feature_dim)  # posterior mean
        # self.P = np.eye(feature_dim)  # posterior precision (inverse covariance)
        self.Pd = np.ones(feature_dim)

        # Discount factor λ in (0,1]; λ=1 disables forgetting
        self.discount = float(discount)

        # Keep a copy of prior diagonal to floor precision
        self._P0_diag = np.ones(feature_dim)

        self.initial_parameters = self._initial_parameters()

    def _initial_parameters(self):
        return {
            "mu": self.mu.copy(),
            "Pd": self.Pd.copy(),
        }

    def select_action(self, actions, feature_vectors):
        feature_vectors = np.array(feature_vectors)

        # Sample theta from the Gaussian prior
        # theta_sample = np.random.multivariate_normal(self.mu, np.linalg.inv(self.P))

        # sampling: Cov = diag(1/Pd)
        theta_sample = self.mu + np.random.normal(size=self.feature_dim) / np.sqrt(self.Pd)

        # Compute the probabilities using the logistic function
        logits = feature_vectors @ theta_sample
        probabilities = 1 / (1 + np.exp(-logits))

        # Select the action with the highest probability
        best_idx = np.argmax(probabilities)
        sampled = {
            "theta": theta_sample,
            "estimated_reward": probabilities[best_idx],
        }
        return actions[best_idx], feature_vectors[best_idx].tolist(), sampled

    def update(self, feature_vector, reward):
        x = np.asarray(feature_vector)

        # --------- Forgetting step (applied BEFORE the new observation) ----------
        if 0 < self.discount < 1:
            # idx = np.diag_indices(self.feature_dim)

            # Decay precision (lose confidence in old info) and floor at prior
            # self.P[idx] *= self.discount
            # self.P[idx] = np.maximum(self.P[idx], self._P0_diag)
            self.Pd = self.Pd * self.discount
            self.Pd = np.maximum(self.Pd, self._P0_diag)

        # Standard Laplace/online-Newton update (diagonal approx)
        sigmoid = 1 / (1 + np.exp(-x @ self.mu))

        # 2) aggiorno la precisione diagonale
        # diag_old = np.diag(self.P).copy()  # s_i^(old)
        # diag_new = diag_old + x**2 * sigmoid * (1 - sigmoid)
        # self.P[np.diag_indices(self.feature_dim)] = diag_new

        self.Pd += (x**2) * sigmoid * (1 - sigmoid)

        # 3) gradiente (solo likelihood, perché gradiente di prior si annulla)
        grad = (sigmoid - reward) * x

        # 4) un passo di Newton coord-by-coord con la precisione aggiornata
        # self.mu = self.mu - grad / diag_new
        self.mu = self.mu - grad / self.Pd

        params = {
            "mu": self.mu.copy(),
            "Pd": self.Pd.copy(),
        }
        return params
