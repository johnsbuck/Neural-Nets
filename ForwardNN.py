import numpy as np
from scipy import optimize

__author__ = 'John Bucknam & Bill Clark'


class ForwardNN(object):
    """A feedforward neural network that capable of regression or classification.

    Requires Numpy to run and takes in ndarray inputs.

    """
    # CONSTANTS
    SUM_OF_SQUARES_TYPE = "Sum of Squares"
    CROSS_ENTROPY_TYPE = "Cross Entropy"

    #Variables
    inputLayerSize = None
    outputLayerSize = None
    hiddenLayerSizes = None
    weight = None

    def __init__(self, layer_sizes, sigfunc=True):
        """Constructor

        Params:
            tuple: Floats; First element is the input layer, last element is the output layer,
            and every other layer acts as an input layer
            sigfunc: Boolean; When true, Neural Network will use sgmoid function, otherwise tanh (Default: True)
        """
        # Define each layer
        self.inputLayerSize = layer_sizes[0]
        self.outputLayerSize = layer_sizes[len(layer_sizes) - 1]
        self.hiddenLayerSizes = layer_sizes[1:len(layer_sizes) - 1]

        self.is_sigmoid = sigfunc

        # Set each weight depending on the number of each layer being paired
        self.weight = []

        # If there are hidden layers
        if len(self.hiddenLayerSizes) != 0:
            # Add random weights from input layer to hidden layer 0
            self.weight.append(np.random.randn(self.inputLayerSize, self.hiddenLayerSizes[0]))
            # For each hidden layer
            for i in range(len(self.hiddenLayerSizes) - 1):
                # Add random weights between each hidden layer
                self.weight.append(np.random.randn(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i + 1]))
            # Add random weights between the last hidden layer and output layer
            self.weight.append(
                np.random.randn(self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize))
        else:
            # Add random weights between the input layer and output layer
            self.weight.append(np.random.randn(self.inputLayerSize, self.outputLayerSize))

    def forward(self, input_matrix):
        """Feeds the input forward through the neural network

        Returns:
             ndarray: Output of our inputted matrix that is [n, 10] (n being the # of inputs)
        """
        # Values taken from each set of weights * input summed together
        self.inputSum = []
        # The inputSums inserted into the threshold function (sigmoid)
        self.threshold = []

        # Append each inputSum value matrix
        self.inputSum.append(np.dot(input_matrix, self.weight[0]))
        for i in range(len(self.weight) - 1):
            # Append each A value from the sigmoid(Z)
            self.threshold.append(self.thresh_func(self.inputSum[i]))
            self.inputSum.append(np.dot(self.threshold[i], self.weight[i + 1]))
        y_hat = self.thresh_func(self.inputSum[len(self.inputSum) - 1])
        return y_hat

    @staticmethod
    def sigmoid(z):
        """Static Sigmoid Function

        Params:
            ndarray: floats

        Returns:
            ndarray: floats
        """
        return 1 / (1 + np.exp(-(z.clip(-100, 100))))

    def sigmoid_prime(self, z):
        """Derivative of the Sigmoid Function

        Params:
            ndarray: floats

        Returns:
            ndarray: floats
        """
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    @staticmethod
    def tanh(z):
        """Static Tanh Function

        Params:
            ndarray: floats

        Returns:
            ndarray: floats
        """
        return np.tanh(z)

    def tanh_prime(self, z):
        """Static Tanh Derivative Function

        Params:
            ndarray: floats

        Returns:
            ndarray: floats
        """
        return 1 - np.power(self.tanh(z), 2)

    def thresh_func_type(self):
        if self.is_sigmoid:
            return "Sigmoid"
        return "Tanh"

    def thresh_func(self, z):
        if self.is_sigmoid:
            return self.sigmoid(z)
        return self.tanh(z)

    def thresh_func_prime(self, z):
        if self.is_sigmoid:
            return self.sigmoid_prime(z)
        return self.tanh_prime(z)

    def cost_function(self, x, y):
        """The cost function that compares the expected output with the actual output for training

        Params:
            ndarray: Input; Should match [n, InputLayerSize] where n is the number of inputs >= to 1.
            ndarray: Output; Should match [n, OutputLayerSize] where n is the number of outputs >= to 1.

            The n value for input and output should be the same.

        Returns:
            None: If invalid training data
            float: Cost of the training data
        """
        y_hat = self.forward(x)
        if y.shape[0] == x.shape[0]:
            type = self.cost_function_type(x, y)
            if type == self.SUM_OF_SQUARES_TYPE:
                return self._sum_of_squares(y, y_hat)
            elif type == self.CROSS_ENTROPY_TYPE:
                return self._cross_entropy(y, y_hat)
        return None

    def cost_function_type(self, x, y):
        if y.shape[1] == 1:
            return self.SUM_OF_SQUARES_TYPE
        elif y.shape[1] > 1:
            return self.CROSS_ENTROPY_TYPE
        else:
            return None

    def _cross_entropy(self, y, y_hat):
        """A type of cost function used for classification greater than 2.

        Returns:
            float: Cost of current neural network.
        """
        return -1 * np.sum(np.multiply(y, self.safe_log(y_hat)) + np.multiply(1 - y, self.safe_log(1 - y_hat))) / y.shape[0]

    @staticmethod
    def safe_log(x, min_val=0.0000000001):
        """A log function used to cap ndarray x when having low or nan elements

        Returns:
            ndarray: Natural log of x matrix
        """
        return np.log(x.clip(min=min_val))

    @staticmethod
    def _sum_of_squares(y, y_hat):
        """A cost function used for regression or binary classification

        Returns:
            float: Cost of current neural network.
        """
        return np.sum(np.power(y - y_hat, 2)) / y.shape[0]

    def cost_function_prime(self, x, y):
        """The derivative of the cost function

        Uses the derived cost function to retrieve the gradient for each component

        Returns:
            list: ndarray; Returns a list of matricies where each matrix represent a specific component
        """
        y_hat = self.forward(x);

        delta = []
        derived = []
        if len(self.weight) > 1:
            # Derivative of cost function * derivative of threshold function(z)
            delta.append(np.multiply(-(y-y_hat), self.thresh_func_prime(self.inputSum[len(self.inputSum) - 1])))
            derived.append(np.dot(self.threshold[len(self.threshold) - 1].T, delta[len(delta) - 1]))

            # Loop for each set of weights
            for i in range(2, len(self.inputSum)):
                delta.append(np.array(np.dot(delta[len(delta) - 1], self.weight[len(self.weight) - i + 1].T)) *
                             np.array(self.thresh_func_prime(self.inputSum[len(self.inputSum) - i])))
                derived.append(np.dot(self.threshold[len(self.threshold) - i].T, delta[len(delta) - 1]))

            # Final set of weights with input
            delta.append(np.array(np.dot(delta[len(delta) - 1], self.weight[1].T)) *
                         np.array(self.thresh_func_prime(self.inputSum[0])))
            derived.append(np.dot(x.T, delta[len(delta) - 1]))
        else:
            # Derivative of cost function * derivative of threshold function(z)
            delta.append(np.multiply(-(y-y_hat), self.thresh_func_prime(self.inputSum[len(self.inputSum) - 1])))
            derived.append(np.dot(x.T, delta[len(delta) - 1]))
        return derived


    def compute_gradients(self, x, y):
        """Returns the gradients from each layer of computation

        Returns:
            ndarray: 1-D array of floats containing the gradient values
        """
        # Obtains the derived costs over each derived weight set
        derived = self.cost_function_prime(x, y)

        # Unravels the derived cost for each set of weights and concatenates them all together
        params = derived[len(derived) - 1].ravel();

        # Concatenates the gradients
        for i in range(len(derived) - 1):
            params = np.concatenate((params.ravel(), derived[len(derived) - 2 - i].ravel()))
        return params

    def get_params(self):
        """Returns the weights in a 1-D array

        Returns:
            ndarray: 1-D array of floats containing the weights. Size of array is [1, n]
                        where n = InputLayerSize * HiddenLayerSize[0] +
                                    range(0,lastHiddenLayer - 1) for HiddenLayerSize[i-1] * HiddenLayerSize[i+1] +
                                    HiddenLayerSize[lastHiddenLayer] * OutputLayerSize
        """
        params = None
        for i in range(len(self.weight)):
            if params is None:
                params = self.weight[i].ravel()
            elif len(params.shape) == 1:
                params = np.concatenate((params.ravel(), self.weight[i].ravel()))
            else:
                params = np.concatenate((params.ravel(), self.weight[i].ravel()), axis=1)
        if len(params.shape) == 2 and params.shape[0] == 1:
            return params.T
        return params

    def set_params(self, params):
        """Sets the weights of the Neural Network from a 1-D array

        Params:
            ndarray: 1-D array of floats that are the weights. Size of array is [1, n]
                        where n = InputLayerSize * HiddenLayerSize[0] +
                                    range(0,lastHiddenLayer - 1) for HiddenLayerSize[i-1] * HiddenLayerSize[i+1] +
                                    HiddenLayerSize[lastHiddenLayer] * OutputLayerSize
        """
        # Starting position of first set of weights
        hiddenStart = 0
        if len(self.weight) > 1:
            # Ending position of first set of weights
            hiddenEnd = self.hiddenLayerSizes[0] * self.inputLayerSize
            # Sets the first set of weights
            self.weight[0] = np.reshape(params[hiddenStart:hiddenEnd], (self.inputLayerSize, self.hiddenLayerSizes[0]))

            # for each hidden layer
            for layer in range(0, len(self.hiddenLayerSizes) - 1):
                # new start position is the previous end position
                hiddenStart = hiddenEnd
                # set new end position
                hiddenEnd = hiddenStart + self.hiddenLayerSizes[layer] * self.hiddenLayerSizes[layer + 1]
                # Sets the set of weights to weight list
                self.weight[layer + 1] = np.reshape(params[hiddenStart:hiddenEnd],
                                           (self.hiddenLayerSizes[layer], self.hiddenLayerSizes[layer + 1]))
            # Setting the final set of weights to output
            hiddenStart = hiddenEnd
            hiddenEnd = hiddenStart + self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1] * self.outputLayerSize
            self.weight[len(self.weight) - 1] = np.reshape(params[hiddenStart:hiddenEnd],
                                            (self.hiddenLayerSizes[len(self.hiddenLayerSizes) - 1], self.outputLayerSize))
        else:
            hiddenEnd = self.inputLayerSize * self.outputLayerSize
            self.weight[0] = np.reshape(params[hiddenStart:hiddenEnd], (self.inputLayerSize, self.outputLayerSize))


class Trainer(object):
    def __init__(self, N):
        """Constructor

        Params:
            NeuralNetwork: Sets neural network to be trained.
        """
        self.neural_net = N

    def cost_function_wrapper(self, params, x, y):
        """Used to set the parameters of the Neural Network being trained

        Returns:
            float: Cost of the current Neural Network
            ndarray: 1-D array of gradients
        """
        # Sets the parameters
        self.neural_net.set_params(params)
        # Gets the cost
        cost = self.neural_net.cost_function(x, y)
        # Obtain the derived cost for each derived weights
        grad = self.neural_net.compute_gradients(x, y)

        return cost, grad

    def train(self, x, y):
        """Trains the Neural Network to fit the training data

        Uses the training method to minimize the cost function and set the weights of the Neural Network.

        Params:
            ndarray: Input value for the Neural Network
            ndarray: Expected output value from the Neural Network
        """
        # Parameters of weights from the Neural Network
        params = self.neural_net.get_params()

        # Options: maximum # of iterations and show information display after minimizing
        opt = {'maxiter': 200, 'disp': False}

        # jac: Jacobian (Defines that cost_function_wrapper returns gradients)
        # BFGS: Uses BFGS method of training and gradient descent
        # callback: Return values acts as parameter for set_params
        optimize.minimize(self.cost_function_wrapper, params, jac=True, method='BFGS',
                          args=(x, y), options=opt, callback=self.neural_net.set_params)
