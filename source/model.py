import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    """

    def __init__(self, input_features, hidden_dim, output_dim):
       
        """
        Initialize the model by setting up linear layers. Use the input parameters to help define the layers of your model.

        :param input_features: 
            the number of input features in your training/test data
        :param hidden_dim: 
            helps define the number of nodes in the hidden layer(s)
        :param output_dim: 
            the number of outputs you want to produce
        """
        
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 12)
        self.fc3 = nn.Linear(12, output_dim)
        self.dropout = nn.Dropout(p=0.25)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
        """
        Perform a forward pass of our model on input features, x.
        
        :param x: 
            A batch of input features of size (batch_size, input_features)
        :return: 
            A single, sigmoid-activated value as output
        """

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.sig(self.fc3(x))
        
        return x
    