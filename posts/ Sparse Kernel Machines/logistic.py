import torch
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand(X.size()[1])
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        return torch.where(scores >= 0, torch.tensor(1.0), torch.tensor(0.0))


class LogisticRegression(LinearModel):

    def loss(self, X, y):
        """
        Computes the binary cross-entropy loss for logistic regression using sigmoid activation.

        This loss measures the difference between predicted probabilities and true binary labels.
        The sigmoid function converts raw model scores to probabilities. To ensure numerical stability,
        the output of the sigmoid is clamped to avoid exactly 0 or 1, preventing undefined values
        in the logarithm computation (log(0) = NaN or log(1) = inf).

        Args:
            X (torch.Tensor): feature matrix of shape (n, p), where N is number of samples.
            y (torch.Tensor): Binary target labels of shape (n,) with values 0 or 1.

        Returns:
            torch.Tensor: Mean binary cross-entropy loss across the batch.
        """
        scores = self.score(X)
        sigmoids = torch.sigmoid(scores)
        sigmoids = torch.clamp(sigmoids, min=1e-7, max=1-1e-7)
        loss_vector = (-y * torch.log(sigmoids)) - ((1-y) * torch.log(1-sigmoids))
        return loss_vector.mean()
    
    def grad(self, X, y):
        """
        Computes the gradient of the binary cross-entropy loss with respect to model weights.

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n,).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the model's weights, shape (p,).
        """
        scores = self.score(X)
        sigmoids = torch.sigmoid(scores)
        error = sigmoids - y
        return  (X.T @ error) / X.shape[0]

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model
        self.w_prev = None

    def step(self, X, y, alpha, beta):
        """
        Performs one gradient descent step to update the model's weights.

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n,) 
            alpha (float): Learning rate 
            beta (float): Momentum factor (0 = no momentum).
        """
     
        if self.model.w is None:
            self.model.score(X)  # initializes self.model.w

        if self.w_prev is None:
            self.w_prev = self.model.w.clone()

        grad = self.model.grad(X, y)
        w_new = self.model.w - alpha * grad + beta * (self.model.w - self.w_prev)

        # Update weights and previous weights
        self.w_prev = self.model.w.clone()
        self.model.w = w_new
        
class KernelLogisticRegression(LinearModel):
    
    def __init__(self, kernel_fn, lam, gamma):
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma
        self.a = None
        self.a_prev = None
        self.X_train = None

    def score(self, X, recompute_kernel):
        if recompute_kernel == False:
            raise ValueError("Model has not been trained or parameters not initialized.")
        K = self.kernel_fn(X, self.X_train, self.gamma)  # [n_test, n_train]
        return (K @ self.a).squeeze()  # [n_test]

    def fit(self, X, y, m_epochs, lr, beta=0.0):
        m = X.shape[0]
        self.X_train = X
        self.a = torch.zeros((m, 1), dtype=X.dtype)
        self.a_prev = torch.zeros_like(self.a)

        K = self.kernel_fn(X, X, self.gamma)  # shape: [m, m]
        y = y.view(-1, 1)  # Ensure y is column vector

        for _ in range(m_epochs):
            s = K @ self.a  # shape: [m, 1]
            sig = torch.sigmoid(s)

            # Gradient of logistic loss w.r.t. a
            grad = K.T @ (sig - y) / m + self.lam * self.a   

            # Momentum update
            a_new = self.a - lr * grad + beta * (self.a - self.a_prev)
            self.a_prev = self.a.clone()
            self.a = a_new

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        s = self.score(X, recompute_kernel=True).unsqueeze(1)
        return torch.sigmoid(s)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        return (self.predict_proba(X) >= 0.5).float()