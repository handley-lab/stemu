from tensorflow import keras  # TensorFlow's Keras API for building deep learning models
from flax import nnx         # Flax, a JAX-based neural network library
import torch.nn as nn        # PyTorch's neural network module

# Base class for defining common model properties
class basemodel():
    def __init__(self, input_shape, output_shape, hidden_structure, activation):
        """
        Initialize the base model parameters.
        Args:
            input_shape (int): Number of input features.
            output_shape (int): Number of output features.
            hidden_structure (list): List defining hidden layer sizes.
            activation (str): Activation function name.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_structure = hidden_structure
        self.activation = activation
    
    def build(self):
        """Placeholder method for building models, to be 
        implemented by subclasses."""
        print("Not implemented.")

# TensorFlow model class inheriting from the base model
class tfmodel(basemodel):
    def __init__(self, input_shape, output_shape, 
                 hidden_structure, activation='tanh'):
        """
        Initialize the TensorFlow model.
        Args:
            input_shape (int): Number of input features.
            output_shape (int): Number of output features.
            hidden_structure (list): List defining hidden layer sizes.
            activation (str): Activation function name.
        """
        super().__init__(input_shape, output_shape, hidden_structure, activation)
    
    def build(self):
        """
        Build a sequential TensorFlow model.
        Returns:
            keras.Model: Constructed TensorFlow model.
        """
        model = keras.models.Sequential(
            [keras.layers.Dense(self.input_shape)]  # Input layer
            + [keras.layers.Dense(self.hidden_structure[0], activation=self.activation)]  # First hidden layer
            + [keras.layers.Dense(hs, activation=self.activation) for hs in self.hidden_structure[1:]]  # Additional hidden layers
            + [keras.layers.Dense(self.output_shape, activation='linear')]  # Output layer with linear activation
        )
        
        return model

# JAX model class inheriting from the base model
class jaxmodel(basemodel):
    def __init__(self, input_shape, output_shape, hidden_structure, activation='tanh', rngs=nnx.Rngs(0)):
        """
        Initialize the JAX model.
        Args:
            rngs (nnx.Rngs): Random number generator seed.
            input_shape (int): Number of input features.
            output_shape (int): Number of output features.
            hidden_structure (list): List defining hidden layer sizes.
            activation (str or nnx activation): Activation function name or nnx activation function.
        """
        super().__init__(input_shape, output_shape, hidden_structure, activation)
        
        self.rngs = rngs  # JAX random seed
        
        # Assign activation functions based on input
        if activation == 'tanh':
            self.activation = nnx.tanh
        elif activation == 'relu':
            self.activation = nnx.relu
        elif activation == 'sigmoid':
            self.activation = nnx.sigmoid
        
    def build(self):
        """
        Build a sequential JAX model.
        Returns:
            nnx.Sequential: Constructed JAX model.
        """
        hs = []
        for i in range(len(self.hidden_structure)):
            if i >= 1:
                hs.append(nnx.Linear(self.hidden_structure[i - 1], self.hidden_structure[i], rngs=self.rngs))  # Linear layer
                hs.append(self.activation)  # Add activation

        model = nnx.Sequential(
            [nnx.Linear(self.input_shape, hs[0].in_features, rngs=self.rngs)]  # Input layer
            + hs  # Hidden layers
            + [nnx.Linear(hs[-2].out_features, self.output_shape, rngs=self.rngs)]  # Output layer
        )
        
        return model

# PyTorch model class inheriting from the base model
class pytorchmodel(basemodel):
    def __init__(self, input_shape, output_shape, hidden_structure, activation='tanh'):
        """
        Initialize the PyTorch model.
        Args:
            input_shape (int): Number of input features.
            output_shape (int): Number of output features.
            hidden_structure (list): List defining hidden layer sizes.
            activation (str or nn activation): Activation function name or nn activation function.
        """
        super().__init__(input_shape, output_shape, hidden_structure, activation)
        
        # Assign PyTorch activation functions based on input
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        
    def build(self):
        """
        Build a sequential PyTorch model.
        Returns:
            nn.Sequential: Constructed PyTorch model.
        """
        # Create a list of all layer sizes including input and output
        layer_nodes = [self.input_shape] + self.hidden_structure + [self.output_shape]
        
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))  # Linear layer
            if i < len(layer_nodes) - 2:
                layers.append(self.activation)  # Add activation except on the final layer
        
        model = nn.Sequential(*layers)  # Create a sequential container
        return model
