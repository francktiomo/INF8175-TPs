import nn
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
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        batch_size = 1
        while True:
            updated = False
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    updated = True
                    self.w.update(x, nn.as_scalar(y))
            if not updated:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.num_layers = 3
        self.batch_size = 1
        self.learning_rate = 0.001
        self.w = [nn.Parameter(1, 10), nn.Parameter(10, 10), nn.Parameter(10, 1)]
        self.b = [nn.Parameter(1, 10), nn.Parameter(1, 10), nn.Parameter(1, 1)]


    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        h = x
        for i in range(self.num_layers):
            if i == self.num_layers - 1: # Not applying ReLU for the last layer
                h = nn.AddBias(nn.Linear(h, self.w[i]), self.b[i])
            else:
                h = nn.ReLU(nn.AddBias(nn.Linear(h, self.w[i]), self.b[i]))
        return h

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        while True:
            has_errors = False
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                if nn.as_scalar(loss) > 0.02:
                    has_errors = True
                    while nn.as_scalar(loss) > 0.02:
                        grad_w_0, grad_b_0, grad_w_1, grad_b_1, grad_w_2, grad_b_2 = nn.gradients(loss, [self.w[0], self.b[0], self.w[1], self.b[1], self.w[2], self.b[2]])
                        self.w[0].update(grad_w_0, -self.learning_rate)
                        self.b[0].update(grad_b_0, -self.learning_rate)
                        self.w[1].update(grad_w_1, -self.learning_rate)
                        self.b[1].update(grad_b_1, -self.learning_rate)
                        self.w[2].update(grad_w_2, -self.learning_rate)
                        self.b[2].update(grad_b_2, -self.learning_rate)
                        loss = self.get_loss(x, y)
            if not has_errors:
                break


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
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.num_layers = 3
        self.batch_size = 10
        self.validation_treshhold = 0.97
        self.learning_rate = 0.01

        self.w = [nn.Parameter(784, 200), nn.Parameter(200, 200), nn.Parameter(200, 10)]
        self.b = [nn.Parameter(1, 200), nn.Parameter(1, 200), nn.Parameter(1, 10)]

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        h = x
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                h = nn.AddBias(nn.Linear(h, self.w[i]), self.b[i])
            else:
                h = nn.ReLU(nn.AddBias(nn.Linear(h, self.w[i]), self.b[i]))
        return h

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        while True:
            acceptable_loss = True
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                if nn.as_scalar(loss) > 0.02:
                    acceptable_loss = False
                    grad_w_0, grad_b_0, grad_w_1, grad_b_1, grad_w_2, grad_b_2 = nn.gradients(loss, [self.w[0], self.b[0], self.w[1], self.b[1], self.w[2], self.b[2]])
                    self.w[0].update(grad_w_0, -self.learning_rate)
                    self.b[0].update(grad_b_0, -self.learning_rate)
                    self.w[1].update(grad_w_1, -self.learning_rate)
                    self.b[1].update(grad_b_1, -self.learning_rate)
                    self.w[2].update(grad_w_2, -self.learning_rate)
                    self.b[2].update(grad_b_2, -self.learning_rate)
                    loss = self.get_loss(x, y)
            if acceptable_loss or dataset.get_validation_accuracy() >= self.validation_treshhold:
                break
    
