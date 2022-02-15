## CENG 499 - HOMEWORK 1 README FILE

This is a 10-class classifier of the CIFAR10 dataset.

This file is to explain the source codes and how to run with different options.

## FILES
1. classes.py: This file contains the k-layered networks defined whose hyperparameters are provided as arguments. Only 1, 2, 3-Layered networks are implemented.

2. ml_1.py: This file contains the overall source code to train and test the neural networks. How to run this is further explained in detail in the following sections in this document. 
It performs either a grid search to tune the hyperparameters or trains and tests the model with hyperparameter configuration taken as command-line arguments, details of which are 
further explained in the relative sections, as well. 
The dataset is first made sure to be loaded and verified and split into training, validation, and test sets, which are all mutually exclusive. When run, it trains the model with either 
user-specified hyperparameters or with a configuration of hyperparameters in the grid search, and tests. Upon training, it prints the current hyperparameters, the validation loss, 
validation accuracy, and training loss on screen for each epoch. Then, it invokes the test phase, which prints the test accuracy.

## GRID SEARCH
When run without explicit command-line arguments, the code performs a grid search to optimize the hyperparameters. 
This is performed through multiple iterations for each k-layered network exclusively, as they all had different types of hyperparameters needed. 

One can find the validation accuracy and loss values for most of the hyperparameter configurations used in the Report. 
As the run time is really high to run for all the grid searches defined, some configurations have not been tried. However, one can easily train with a hyperparameter configuration 
that is not reported by giving command-line arguments.

For 1-Layered Network the following hyperparameters are used:
batch_sizes = [64, 32], learning_rates= [0.01, 0.001, 0.0005], optimizer = ["SGD", "Adam"], epochs = [25,50,75,100].

``` 
		model = OneLayer().to(device)
		for batch_size in batch_sizes:
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
			val_loader = DataLoader(val_set, batch_size=batch_size)
			test_loader = DataLoader(test_set, batch_size=batch_size)
			for lr in learning_rates:
				for opt in optimizer:
					print("opt " ,opt)	
					for epoch in epochs:
						#train(num_epochs, learning_rate, train_loader, val_loader, test_loader, model, optimizer, momentum)
						print()
						print("curr learning_rate: ", lr)
						print("optimizer: ", optimizer)
						print()
						train(epoch, lr, train_loader, val_loader, test_loader, model, opt)
```
Hyperparameters tried for 2-Layered Network are: 

number of epochs: [25,50,75,100], 
learning rates: [0.01,0.001,0.0005], 
batch size = 64, 
dropout rate: [0.2, 0.5], 
activation functions: ["ReLU","hardswish"], 
number of neurons: [512,256]


Hyperparameters tried for 3-Layered Network are: 

number of epochs: [25,50,75,100], 
learning rates: [0.01,0.001,0.0005], 
batch size: 64, 
dropout rate: [0.2, 0.5], 
activation functions: ["ReLU","hardswish"], 
number of neurons in the first layer: [512, 256], 
number of neurons in the second layer: [256, 128]

And the grid search implementation for 2 and 3-Layered Networks can be found in the ml_1.py file, as well. (starting at lines 262, and 298 respectively).


## HOW TO RUN IN GRID SEARCH MODE
When run with no explicit argument, it runs in grid search mode.
``` python ml_1.py  ```


## Run with User Specified Hyperparameters

One has to explicitly run the code with ``` --define_hyperparam 1 ``` flag to run in this mode. 
Example runs are provided below.

Users can specify the following hyperparameters with the argument specified for each. And, when run in this mode, if any of the hyperparameters is not set, the default values corresponding 
to each (specified below in parentheses for each) are set.

Number of layers in the network: --num_layers (default set to 3),

Number of neurons in the first layer: --num_neurons1 (default set to 512),

Number of neurons in the second layer: --num_neurons2 (default set to 256),

Number of epochs: --num_epochs (default set to 100),

Learning Rate: --lr (default set to 0.001),

Batch size: --batch_size (default set to 64),

Optimizer: --optimizer (default set to Adam),

Momentum: --momentum (default set to 0.9),

Activation function for the first layer: --activation1 (default set to ReLU),

Activation function for the second layer: --activation2 (default set to ReLU),

Drop out rate: --dropout (default set to 0.2)


If any of the arguments are not specified, they are taken as their default value. 

One must not provide a semantically meaningless argument configuration. For example, for a 2-Layered network, providing an activation function or the number of neurons 
for the second layer is a faulty configuration that will not yield any meaningful result. 
Furthermore, only 1, 2, and 3-Layered Networks are implemented, so one is not to provide arguments as --num_layers > 3


```
usage: ml_args.py [-h] [--define_hyperparam DEFINE_HYPERPARAM] [--num_layers NUM_LAYERS] [--num_neurons1 NUM_NEURONS1] [--num_neurons2 NUM_NEURONS2] [--num_epochs NUM_EPOCHS] [--lr LR]
                  [--batch_size BATCH_SIZE] [--optimizer OPTIMIZER] [--momentum MOMENTUM] [--activation1 ACTIVATION1] [--activation2 ACTIVATION2] [--dropout DROPOUT]
```


Example runs:

For 1-Layered Network:

``` python ml_1.py --define_hyperparam 1 --num_layers 1 --num_epochs 25 ``` 

(here learning rate is taken to be the default value, ie, 0.001)

For 2-Layered Network:

``` python ml_1.py --define_hyperparam 1 --num_layers 2 --lr 0.001 --num_neurons1 512 --activation1 hardswish --dropout 0.5 --num_epochs 25  ``` 

For 3-Layered Network:
``` python ml_1.py --define_hyperparam 1 --num_layers 3 --num_neurons2 512 --activation1 hardswish --activation2 tanh --num_epochs 25 ```

``` python ml_1.py --define_hyperparam 1 --num_layers 3 --num_neurons1 512 --num_neurons2 256 --activation1 hardswish --activation2 tanh --num_epochs 25 --lr 0.0005 --dropout 0.8 ```



The following are the run command for each k-layered network with the best performance:
For 1-layered:
``` python ml_1.py --define_hyperparam 1 --num_layers 1 --batch_size 32 --lr 0.0005  --num_epochs 25 ``` 

For 2-layered:
``` python ml_1.py --define_hyperparam 1 --num_layers 2 --batch_size 64 --lr 0.001 --num_neurons1 512 --activation1 hardswish --dropout 0.5  --num_epochs 25 ```

For 3-layered:
``` python ml_1.py --define_hyperparam 1 --num_layers 3 --batch_size 64 --lr 0.001 --num_neurons1 512 --num_neurons1 256 --activation1 tanh --activation2 tanh --dropout 0.2  --num_epochs 50``` 


