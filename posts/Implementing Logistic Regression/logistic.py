import torch

torch.manual_seed(1234)

def perceptron_data(n_points = 300, noise = 0.2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,2))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)

    # convert y from {0, 1} to {-1, 1}
    y = 2*y - 1

    return X, y

X, y = perceptron_data(n_points = 300, noise = 0.2)


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

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy way to make this conversion is: 
        
        y_ = 2*y - 1
        """
        y_ = 2 * y - 1
        scores = self.score(X)
        return (scores * y_ < 0).float().sum() / scores.numel()

    def grad(self, X, y):
        score_i = self.score(X) 
        misclassified = score_i * (2*y - 1) < 0 
        return (-1 * misclassified) * ((2*y - 1) * X)


class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        self.model.loss(X, y)
        self.model.w -= self.model.grad(X,y)[0]

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
        
