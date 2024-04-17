import nn
import numpy as np
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)
                

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1
        

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        converged = False

        while not converged:
            converged = True 

            # Iterate over each data point in the dataset
            for x, y in dataset.iterate_once(batch_size=1):
                prediction = self.get_prediction(x)
                target = nn.as_scalar(y)

                # Update weights if prediction is incorrect
                if prediction != target:
                    # Calculate the update direction based on the target label
                    update_direction = x.data if target == 1 else -x.data
                    # Update the weights
                    self.w.update(nn.Constant(update_direction), 1.0)
                    converged = False 

            if converged:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize model parameters
        self.w1 = nn.Parameter(1, 100) 
        self.b1 = nn.Parameter(1, 100) 
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, 1)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1)) 
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2)) 
        return nn.AddBias(nn.Linear(l2, self.w3), self.b3)

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        epochs = 100
        learning_rate = 0.1
        loss_threshold = 0.02

        parameters = [self.w1, self.b1, self.w2, self.b2]

        for _ in range(epochs):
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < loss_threshold:
                break

            for x, y in dataset.iterate_once(batch_size=1):
                curr_loss = self.get_loss(x, y)
                grads = nn.gradients(curr_loss, [self.w1, self.b1, self.w2, self.b2])

                for param in parameters:
                    param.update(grads[parameters.index(param)], -learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize model parameters
        self.w1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, 10)
        self.b3 = nn.Parameter(1, 10)
        
    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        return nn.AddBias(nn.Linear(l2, self.w3), self.b3)
    

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)
    

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        epochs = 250
        learning_rate = 0.5
        accuracy_threshold = 0.975

        parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        for _ in range(epochs):
            for x, y in dataset.iterate_once(batch_size=1500):
                current_accuracy = dataset.get_validation_accuracy()
                
                if current_accuracy > accuracy_threshold:
                    break

                curr_loss = self.get_loss(x, y)
                grads = nn.gradients(curr_loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
               
                for param in parameters:
                    param.update(grads[parameters.index(param)], -learning_rate)
