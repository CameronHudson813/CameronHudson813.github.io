import torch
import numpy as np

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
            self.w = torch.rand((X.size()[1]))
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
    # #Normal approach
    # def grad(self, X, y):
    #     score_i = self.score(X) 
    #     misclassified = score_i * (2*y - 1) < 0 

    #     return (-1 * misclassified) * ((2*y - 1) * X)
    # Minibatch implementation
    def grad(self, X, y):
        # I believe my x is an array within an array
        y_ = 2*y - 1
        scores = self.score(X)
        misclassified = (scores * y_ < 0).float()
        grads = (misclassified * y_).unsqueeze(1) * X
        return grads.sum(dim=0)
        


class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    # def step(self, X, y):
    #     """
    #     Compute one step of the perceptron update using the feature matrix X 
    #     and target vector y. 
    #     """
    #     self.model.loss(X, y)
    #     self.model.w -= self.model.grad(X,y)[0]
    # minibatch implementation
    def step(self, X, y, alpha, k):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        self.model.loss(X, y)
        self.model.w -= (alpha/k) * self.model.grad(X,y) #possibly need to keep this