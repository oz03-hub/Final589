import numpy as np
from tqdm import tqdm

def print_matrix(matrix: np.ndarray):
    for row in matrix:
        print(f"\t\t{row}")

class NeuralNet:
    def __init__(self, layers: list[int], step_size: float, lambda_reg: float, batch_size: int = 150, epochs: int = 10, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.step_size = step_size
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.layers = layers
        self.thetas = []
        self.epochs = epochs
        self._initialize_layers(layers)

    def model_str(self):
        return f"NN Layers: {self.layers}, Step size: {self.step_size}, Lambda: {self.lambda_reg}, Batch size: {self.batch_size}, Epochs: {self.epochs}"

    def _initialize_layers(self, layers: list[int]):
        self.thetas = []
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            input_size = layers[i] + 1 # adding bias term
            output_size = layers[i+1]

            self.thetas.append(np.random.normal(loc=0.0, scale=1.0, size=(input_size, output_size)))
    
    def _activation(self, z: np.ndarray) -> np.ndarray:
        # sigmoid activator
        return 1 / (1 + np.exp(-z))


    def _forward_pass(self, x: np.ndarray, verbose: bool = False) -> np.ndarray:
        a = np.insert(x.reshape(1, -1), 0, 1, axis=1) # adding bias term
        activations = [a]

        if verbose:
            print(f"\t\ta1: {a}")

        for l in range(len(self.thetas) - 1):
            theta = self.thetas[l]
            z = np.matmul(activations[-1], theta)
            a = self._activation(z)
            a = np.insert(a, 0, 1, axis=1) # adding bias term
            activations.append(a)

            if verbose:
                print(f"\t\tz{l+2}: {z}")
                print(f"\t\ta{l+2}: {a}")
        
        # last layer
        theta = self.thetas[-1]
        z = np.matmul(activations[-1], theta)
        a = self._activation(z)
        activations.append(a)

        if verbose:
            print(f"\t\tz{len(self.thetas)+1}: {z}")
            print(f"\t\ta{len(self.thetas)+1}: {a}")
            print(f"\t\tf(x): {a}")

        return activations

    def _backward_pass(self, activations: list[np.ndarray], y: np.ndarray):
        delta_L = activations[-1] - y # delta of all output neurons
        deltas = [delta_L]

        for l in range(len(self.layers) - 2, -1, -1):
            theta = self.thetas[l].T
            delta_next = deltas[-1]

            a = activations[l]
            delta = np.matmul(delta_next, theta) * a * (1 - a) # theta.T . delta_next .* a .* (1- a)
            delta = delta[:, 1:] # removing bias term
            deltas.append(delta)

        deltas.reverse()
        return deltas

    def _compute_cost(self, z: np.ndarray, y: np.ndarray):
        e = 1e-8
        J = -y * np.log(z + e) - (1 - y) * np.log(1 - z + e)
        J = J.sum()
        return J
    
    def _compute_reg_term(self):
        reg_term = 0
        for theta in self.thetas:
            theta_reg = theta.copy()
            theta_reg[0, :] = 0
            reg_term += np.sum(theta_reg ** 2)
        return reg_term

    def refit(self, X: np.ndarray, Y: np.ndarray):
        self._initialize_layers(self.layers)
        return self.fit(X, Y)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False, disable_bar: bool = False):
        # perform mini batch fitting
        losses = []
        for epoch in tqdm(range(self.epochs), desc="training", disable=disable_bar):
            reg_term = self._compute_reg_term()

            J = 0
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i+self.batch_size].copy()
                y_batch = Y[i:i+self.batch_size].copy()

                if verbose:
                    print(f"X_batch: {X_batch}")
                    print(f"y_batch: {y_batch}")

                D = [np.zeros(theta.shape) for theta in self.thetas] # store the gradients
                for x, y in zip(X_batch, y_batch):
                    if verbose:
                        print(f"Processing training instance")
                        print(f"\tInput: {x}")

                    activations = self._forward_pass(x, verbose)
                    ji = self._compute_cost(activations[-1], y)
                    J += ji
                    if verbose:
                        print(f"\tPredicted output: {activations[-1]}")
                        print(f"\tExpected output: {y}")
                        print(f"\tCost, J, associated with instance: {ji}")

                    deltas = self._backward_pass(activations, y)
                    if verbose:
                        print(f"\tComputing gradients")
                        print(f"\tDeltas:")
                        for i in range(len(deltas)-1, 0, -1):
                            print(f"\t\tDelta[{i+1}]: {deltas[i]}")

                    for i in range(len(D) - 1, -1, -1):
                        gradients = np.matmul(activations[i].T, deltas[i+1]) # D = D + delta_next . a.T
                        D[i] += gradients
                        if verbose:
                            print(f"\tGradients of Theta{i+1}:")
                            print_matrix(gradients.T)
                    
                    if verbose:
                        print("---")

                if verbose:
                    print(f"\tThe batch has been processed. Computing the average (regularized) gradients:")

                # compute final regularized gradients
                for i in range(len(self.layers) - 2, -1, -1):
                    theta_reg = self.thetas[i].copy()
                    theta_reg[0, :] = 0                    
                    D[i] = (1/len(X_batch)) * (D[i] + self.lambda_reg * theta_reg)
                    
                    if verbose:
                        print(f"\tFinalized average gradients of Theta{i+1}:")
                        print_matrix(D[i].T)
                
                # update weights
                for i in range(len(self.layers) - 2, -1, -1):
                    self.thetas[i] = self.thetas[i] - self.step_size * D[i]
            
            reg_term = (self.lambda_reg / (2 * len(X))) * reg_term
            J /= len(X)
            J += reg_term
            if verbose:
                print(f"Final (regularized) cost, J, based on the complete training set: {J}")
            losses.append(J)

        return losses

    def predict(self, X: np.ndarray):
        predictions = []
        for x in X:
            output = self._forward_pass(x)
            predictions.append(output[-1])
        return predictions

    def loss_against_test(self, X_test: np.ndarray, y_test: np.ndarray):
        predictions = self.predict(X_test)
        J = self._compute_cost(np.array(predictions), y_test)
        reg_term = (self.lambda_reg / (2 * len(X_test))) * self._compute_reg_term()
        J /= len(X_test)
        J += reg_term
        return J
    