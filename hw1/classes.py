import torch
import torch.nn as nn

# Define the ANN

class OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        #layers of the nnet
        self.layer1 = nn.Linear(in_features=32*32, out_features=10)

    
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.layer1(x)
      return x


class TwoLayer(nn.Module):
    def __init__(self, num_layer1, act_func,dropout):
        super().__init__()
        #layers of the nnet
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=10)
        self.act_func = act_func
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x):
      #inp layer
      x = torch.flatten(x, 1)
      
      x = self.layer1(x)
      x = self.batchnorm1(x)      
      if self.act_func == "tanh":
        x = torch.tanh(x)
      elif self.act_func == "ReLU":
        self.act_func = nn.functional.relu(x)
      elif self.act_func == "hardswish":
        self.act_func = nn.functional.hardswish(x)
      x = self.dropout(x)

      #hidden
      x = self.layer2(x)

      return x


class ThreeLayer(nn.Module):
    def __init__(self, num_layer1, num_layer2, act_func1, act_func2, dropout):
        super().__init__()
        #layers of the nnet
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_layer1)

        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_layer2)

        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)

        self.act_func1 = act_func1
        self.act_func2 = act_func2

        self.dropout = nn.Dropout(p=dropout)



    
    def forward(self, x):
      x = torch.flatten(x, 1)

      #hidden layer 1
      x = self.layer1(x)
      x = self.batchnorm1(x)
      if self.act_func1 == "tanh":
        x = torch.tanh(x)
      elif self.act_func1 == "ReLU":
        self.act_func1 = nn.functional.relu(x)
      elif self.act_func1 == "hardswish":
        self.act_func1 = nn.functional.hardswish(x)
      x = self.dropout(x)

      #hidden layer 2
      x = self.layer2(x)
      x = self.batchnorm2(x)
      if self.act_func2 == "tanh":
        x = torch.tanh(x)
      elif self.act_func2 == "ReLU":
        self.act_func2 = nn.functional.relu(x)
      elif self.act_func2 == "hardswish":
        self.act_func2 = nn.functional.hardswish(x)
      x = self.dropout(x)  

      x = self.layer3(x)    

      return x 