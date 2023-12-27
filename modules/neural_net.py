import torch.nn.functional as F 
from typing import Tuple
import torch.nn as nn
import networkx as nx
import numpy as np
import torch 

class BinaryClassifier(nn.Module):
    """
    Binary Classifier class that inherits from the nn.Module class.
    """
    #Here we define the number of classes
    OUTPUT_SIZE = 2
    def __init__(self, input_size: int = 10, hidden_size: int = 6, dropout: float = 0.01):
        """
        Class constructor for a Binary Classifier.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of nodes in the hidden layer.
            dropout (float): Dropout rate after the first layer.
        """
        #Here we call the constructor of the parent class
        super(BinaryClassifier,self).__init__()
        #Here we define the first layer as a linear function
        self.fc1 = nn.Linear(input_size, hidden_size)
        #Here we define the dropout layer
        self.drop1 = nn.Dropout(p=dropout)
        #Here we define the second layer as a linear function
        self.fc2 = nn.Linear(hidden_size, self.OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network i.e. the prediction function.

        Args:
            x (Tensor): Input tensor with shape (num_samples, num_features).

        Returns:
            x (Tensor): Output tensor with shape (num_samples, num_classes).
        
        """
        #Here we apply the first layer, a linear function
        x = self.fc1(x)
        #Then we apply the ReLU function i.e. the activation function that performs the non-linearity
        x = F.relu(x)
        #Then we apply the dropout layer in order to avoid overfitting
        x = self.drop1(x)
        #Finally we apply the second layer, a linear function
        x = self.fc2(x)
        #We return the output
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function that predicts the class of the input data.

        Args:
            x (Tensor): Input tensor with shape (num_samples, num_features).

        Returns:
            x (Tensor): Output tensor with shape (num_samples,).
        """
        #First we set the model to evaluation mode
        self.eval()
        #First we apply the Softmax function to the output in order to obtain the probabilities of each class
        pred = F.softmax(self.forward(x),dim=1)

        #Here we initialize the list of predictions
        ans = []

        #Here we choose the class with more weight. If the first class has more weight than the second one, we choose the first class
        #If the second class has more weight than the first one, we choose the second class
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)

        #Here we return the predictions as a tensor
        return torch.tensor(ans)
    
    def predict_proba(self,x: torch.Tensor) -> np.ndarray:
        """
        Function that predicts the probabilities of each class of the input data.

        Args:
            x (Tensor): Input tensor with shape (num_samples, num_features).

        Returns:
            pred (np.ndarray): Output tensor with shape (num_samples, num_classes).
        """
        #First we set the model to evaluation mode
        self.eval()
        #We apply the Softmax function to the output in order to obtain the probabilities of each class
        pred = F.softmax(self.forward(x),dim=1)

        #Here we return the predictions as a numpy array
        return  pred.detach().numpy()

    
def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, train_x: torch.Tensor, train_y:torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor) -> Tuple[float, float]:
    """
    Function that performs a training step.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_x (Tensor): The training data.
        train_y (Tensor): The training labels.
        val_x (Tensor): The validation data.
        val_y (Tensor): The validation labels.

    Returns:
        loss_train.item() (float): The loss of the training data.
        loss_val.item() (float): The loss of the validation data.
    """
    #Here we set the model to training mode
    model.train()
    #Here we set the gradients to zero to avoid accumulating them
    optimizer.zero_grad()
    #Here we apply the model to the training data
    y_pred_train = model.forward(train_x)
    #Here we compute the loss of the training data
    loss_train = F.cross_entropy(y_pred_train, train_y)
    #We obtain the gradients and perform backpropagation
    loss_train.backward()
    #We adjust the weights
    optimizer.step()

    #Here we set the model to evaluation mode
    model.eval()
    #Here we set the gradients to zero to avoid accumulating them
    with torch.no_grad():
        #Here we apply the model to the validation data
        y_pred_val = model.forward(val_x)
        #Here we compute the loss of the validation data
        loss_val = F.cross_entropy(y_pred_val, val_y)

    return loss_train.item(), loss_val.item()

class Network():
    """
    Class that plots a network with the number of nodes per layer given by the user.
    This class was taken from the course notes of a previous Machine Learning course taken by Miguel A. Sanchez Cortes during his Bachelor's degree in Physics at UNAM.
    We don't take credit for this class and we don't claim it as our own. We use it only for visualization purposes.
    """

    def  __init__ (self, sizes: list):
        """
        Class constructor for a Network.
        """
        #Here we define the number of layers
        self.num_layers = len(sizes)
        print("It has", self.num_layers, "layers,")
        #Here we define the number of nodes per layer
        self.sizes = sizes
        print("with the following number of nodes per layer", self.sizes)

    def graph(self):
        """
        Function that plots the network.
        """
        a=[]
        ps={}
        Q = nx.Graph()
        for i in range(self.num_layers):
            Qi=nx.Graph()
            n=self.sizes[i]
            nodos=np.arange(n)
            Qi.add_nodes_from(nodos)
            l_i=Qi.nodes
            Q = nx.union(Q, Qi, rename = (None, 'Q%i-'%i))
            if len(l_i)==1:
                ps['Q%i-0'%i]=[i/(self.num_layers), 1/2]
            else:
                for j in range(len(l_i)+1):
                    ps['Q%i-%i'%(i,j)]=[i/(self.num_layers),(1/(len(l_i)*len(l_i)))+(j/(len(l_i)))]
            a.insert(i,Qi)
        for i in range(len(a)-1):
            for j in range(len(a[i])):
                for k in range(len(a[i+1])):
                    Q.add_edge('Q%i-%i' %(i,j),'Q%i-%i' %(i+1,k))
        nx.draw(Q, pos = ps)
