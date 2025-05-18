import torch
from torch.autograd.functional import hessian
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
    
    def hessian(self, X):
        """
        Computes the Hessian matrix for logistic regression.
        
        The Hessian is given by X^T * D * X where D is a diagonal matrix with 
        entries D_ii = σ(s_i) * (1 - σ(s_i)), where s_i is the score for the ith example.
        
        Args:
            X (torch.Tensor): Input features of shape (n, p).
            
        Returns:
            torch.Tensor: Hessian matrix of shape (p, p).
        """
        scores = self.score(X)
        sigmoids = torch.sigmoid(scores)
        # Calculate σ(s) * (1 - σ(s)) for each example
        diag_elements = sigmoids * (1 - sigmoids)
        
        # Matrix formulation to avoid for-loops:
        # First create X weighted by √(diag_elements)
        X_weighted = X * torch.sqrt(diag_elements).unsqueeze(1)
        
        # Then compute X^T * D * X as (X_weighted)^T * X_weighted
        H = X_weighted.T @ X_weighted
        return H

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model
        self.w_prev = None

    def step(self, X, y, alpha, beta, minibatch, batch_size):
        """
        Performs one gradient descent step to update the model's weights.

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n,) 
            alpha (float): Learning rate 
            beta (float): Momentum factor (0 = no momentum).
        """
        if minibatch:
            k = batch_size
            ix = torch.randperm(X.size(0))[:k]
            X = X[ix,:]
            y = y[ix]
            
        if self.model.w is None:
            self.model.score(X)  # initializes self.model.w

        if self.w_prev is None:
            self.w_prev = self.model.w.clone()

        grad = self.model.grad(X, y)

        if minibatch:
            w_new = self.model.w - (alpha / k) * grad + beta * (self.model.w - self.w_prev)
        else:
            w_new = self.model.w - alpha * grad + beta * (self.model.w - self.w_prev)

        # Update weights and previous weights
        self.w_prev = self.model.w.clone()
        self.model.w = w_new
        
class NewtonOptimizer:
    
    def __init__(self, model):
        """
        Initializes the Newton optimizer.
        
        Args:
            model (LogisticRegression): The logistic regression model to optimize.
        """
        self.model = model
    
    def step(self, X, y, alpha):
        """
        Performs one Newton optimization step to update the model's weights.

        Newton's method uses second-order information (the Hessian) to take a more
        informed step in the direction of the minimum.

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n).
            alpha (float): Learning rate (step size)
        """

        if self.model.w is None:
            self.model.score(X)  # initializes self.model.w
        
        # Compute the gradient
        grad = self.model.grad(X, y)
        
        # Compute the Hessian
        H = self.model.hessian(X)
        # Compute the Newton direction: H^(-1) * grad
        # Using torch.linalg.solve for numerical stability 
        newton_direction = torch.linalg.solve(H, grad)
        
        # Update the weights
        self.model.w = self.model.w - alpha * newton_direction

class AdamOptimizer:
    def __init__(self, model):
        """
        Initializes the Adam optimizer.

        Args:
            model (LogisticRegression): The logistic regression model to optimize.
        """
        self.model = model
        self.m = None
        self.v = None
        self.t = 0  
        self.w_0 = None
    
    def grad(self, X, y, batch_size):
        """
        Computes a stochastic estimate of the gradient using a mini-batch.

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n).
            batch_size (int): Number of samples in the mini-batch.

        Returns:
            torch.Tensor: Gradient estimate of shape (p).
        """
        k = batch_size
        ix = torch.randperm(X.size(0))[:k]

        scores = self.model.score(X[ix,:])
        sigmoids = torch.sigmoid(scores)
        error = sigmoids - y[ix]
        return  (X[ix,:].T @ error) / X[ix,:].shape[0]

    def step(self, X, y, alpha, beta_1, beta_2, eps, batch_size):
        """
        Performs one Adam optimization step to update the model's weights.

        Adam combines momentum and adaptive learning rate techniques for 
        efficient and robust training.

        Update rule:
            m ← β₁ * m + (1 - β₁) * grad
            v ← β₂ * v + (1 - β₂) * grad²
            m̂ ← m / (1 - β₁^t)
            v̂ ← v / (1 - β₂^t)
            w ← w - α * m̂ / (sqrt(v̂) + ε)

        Args:
            X (torch.Tensor): Input features of shape (n, p).
            y (torch.Tensor): Binary target labels of shape (n,).
            alpha (float): Learning rate (e.g., 0.001).
            beta_1 (float): Decay rate for first moment estimate (e.g., 0.9).
            beta_2 (float): Decay rate for second moment estimate (e.g., 0.999).
            eps (float): Small constant for numerical stability (e.g., 1e-8).
            batch_size (int): Mini-batch size for stochastic updates.
        """
        if self.model.w is None:
            self.model.score(X)  # initializes self.model.w

        # Compute mini-batch gradient
        grad = self.grad(X, y, batch_size)

        # Initialize moving averages if needed
        if self.m is None:
            self.m = torch.zeros_like(grad)
        if self.v is None:
            self.v = torch.zeros_like(grad)

        # Update timestep
        self.t += 1

        # Update biased first and second moment estimates
        self.m = beta_1 * self.m + (1 - beta_1) * grad
        self.v = beta_2 * self.v + (1 - beta_2) * (grad ** 2)

        m_hat = self.m / (1 - beta_1 ** self.t)
        v_hat = self.v / (1 - beta_2 ** self.t)

        # Update weights
        self.model.w = self.model.w - alpha * m_hat / (torch.sqrt(v_hat) + eps)