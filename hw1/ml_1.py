import argparse

# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

#import network classes
from classes import OneLayer, TwoLayer, ThreeLayer

# Fix the randomness
seed = 1234
torch.manual_seed(seed)

#set device to gpu if avail
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from torchvision.datasets import CIFAR10
import torchvision.transforms as T

train_transform = T.Compose ([
# can add additional transforms on images
T.ToTensor(), # convert images to PyTorch tensors
T.Grayscale(), # RGB to grayscale
T.Normalize(mean =(0.5 ,) , std=(0.5 ,) ) # n or ma li za ti on
# speeds up the convergence
# and improves the accuracy
])

val_transform = test_transform = T.Compose ([
T.ToTensor (),
T.Grayscale (),
T.Normalize ( mean =(0.5 ,), std=(0.5 ,) )
])

train_set = CIFAR10( root ="CIFAR10", train =True, transform = train_transform, download = True )
test_set = CIFAR10( root ="CIFAR10", train =False, transform = test_transform , download = True )

#splitting train/validation sets
train_set_length = int(0.8 * len(train_set))
val_set_length = len(train_set) - train_set_length
train_set, val_set = random_split(train_set, [train_set_length, val_set_length])




parser = argparse.ArgumentParser()
parser.add_argument(
    '--define_hyperparam',
    type=int,
    default=0,
    help='')
parser.add_argument(
    '--num_layers',
    type=int,
    default=3,
    help='Number of layers in the network')

parser.add_argument(
    '--num_neurons1',
    type=int,
    default=512,
    help='Number of neurons in the first layer')

parser.add_argument(
    '--num_neurons2',
    type=int,
    default=256,
    help='Number of neurons in the second layer')

parser.add_argument(
    '--num_epochs',
    type=int,
    default=100,
    help='Number of epochs')

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning Rate')

parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size')

parser.add_argument(
    '--optimizer',
    type=str,
    default="Adam",
    help='Optimizer')

parser.add_argument(
    '--momentum',
    type=float,
    default=.09,
    help='momentum')

parser.add_argument(
    '--activation1',
    type=str,
    default="ReLU",
    help='Activation function for the first layer')

parser.add_argument(
    '--activation2',
    type=str,
    default="ReLU",
    help='activation function for the second layer')

parser.add_argument(
    '--dropout',
    type=float,
    default=0.2,
    help='Drop out rate')






def train(num_epochs, learning_rate, train_loader, val_loader, test_loader, model, optimizer, momentum=0.9):
	print(num_epochs, learning_rate, train_loader, val_loader, test_loader, model, optimizer, momentum)
	loss_function = nn.CrossEntropyLoss()

	if optimizer == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
	elif optimizer == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	else:
		print("Please enter either SGD or Adam")
		print("as --optimizer SGD OR --optimizer Adam")
		return


	for epoch in tqdm(range(num_epochs)):
	    # Training
	    model.train()
	    accum_train_loss = 0
	    for i, (imgs, labels) in enumerate(train_loader, start=1):
	        imgs, labels = imgs.to(device), labels.to(device)
	        output = model(imgs)
	        loss = loss_function(output, labels)

	        # accumlate the loss
	        accum_train_loss += loss.item()

	        # backpropagation
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    
	    # Validation
	    model.eval()
	    accum_val_loss = 0
	    with torch.no_grad():
	        correct = total = 0
	        for j, (imgs, labels) in enumerate(val_loader, start=1):
	            imgs, labels = imgs.to(device), labels.to(device)
	            output = model(imgs)
	            _, predicted_labels = torch.max(output, 1)
	            accum_val_loss += loss_function(output, labels).item()
	            correct += (predicted_labels == labels).sum()
	            total += labels.size(0)

	        print(f'Validation Accuracy = {100 * correct/total :.3f}%')

	    # print statistics of the epoch
	    print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')

	test(model, test_loader)


def test(model, test_loader):
	# Compute Test Accuracy
	model.eval()
	with torch.no_grad():
	    correct = total = 0
	    for images, labels in test_loader:
	        images, labels = images.to(device), labels.to(device)
	        output = model(images)
	        
	        _, predicted_labels = torch.max(output, 1)
	        correct += (predicted_labels == labels).sum()
	        total += labels.size(0)

	print(f'Test Accuracy = {100 * correct/total :.3f}%')





if __name__ == "__main__":
	args = parser.parse_args()


	## if hyperparameters are given as cmd line args
	if args.define_hyperparam == 1:
		#check args validity
		if args.num_layers != 1 and args.num_layers != 2 and args.num_layers != 3:
			print("Please provide 1, 2, or 3 layers as argument")
			exit()


		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = DataLoader(val_set, batch_size=args.batch_size)
		test_loader = DataLoader(test_set, batch_size=args.batch_size)

		if args.num_layers == 1:
			model = OneLayer().to(device)
			train(args.num_epochs, args.lr, train_loader, val_loader, test_loader, model, args.optimizer, args.momentum)

		elif args.num_layers == 2:
			#(self, num_layer1, act_func,dropout)
			model = TwoLayer(args.num_neurons1, args.activation1, args.dropout).to(device)
			train(args.num_epochs, args.lr, train_loader, val_loader, test_loader, model, args.optimizer, args.momentum)

		elif args.num_layers == 3:
			#(self, num_layer1, num_layer2, act_func1, act_func2, dropout)
			model = ThreeLayer(args.num_neurons1, args.num_neurons2, args.activation1, args.activation2, args.dropout).to(device)
			train(args.num_epochs, args.lr, train_loader, val_loader, test_loader, model, args.optimizer, args.momentum)




	#do a grid search
	else:
		print("Grid Search on 1-Layered Network")

		batch_sizes = [64, 32]
		learning_rates= [0.01, 0.001, 0.0005]
		optimizer = ["SGD", "Adam"]
		epochs = [25,50,75,100]

		model = OneLayer().to(device)
		for batch_size in batch_sizes:
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
			val_loader = DataLoader(val_set, batch_size=batch_size)
			test_loader = DataLoader(test_set, batch_size=batch_size)
			for lr in learning_rates:
				for opt in optimizer:
					for epoch in epochs:
						#train(num_epochs, learning_rate, train_loader, val_loader, test_loader, model, optimizer, momentum)
						print()
						print("curr learning_rate: ", lr)
						print("optimizer: ", opt)
						print()
						train(epoch, lr, train_loader, val_loader, test_loader, model, opt)




		print("Grid search on 2-Layered Network")
		num_epochs = [25,50,75,100]
		learning_rates = [0.01,0.001,0.0005]
		batch_size = 64
		dropout = [0.2, 0.5]
		act_func = ["ReLU","hardswish"]
		nneuron = [512,256]

		#(self, num_layer1, act_func, dropout)
		for lr in learning_rates:
			for num in nneuron:
				n1 = num
				for func in act_func:
					a1 = func
					for d in dropout:	
						for epoch in epochs:
							# Fix the randomness
							seed = 1234
							torch.manual_seed(seed)					
							model = TwoLayer(n1,a1,d).to(device)
							optimizer = torch.optim.Adam(model.parameters(), lr=lr)							
					

							print()
							print("curr learning_rate: ", lr)
							print("num of neurons in the first layer: ", n1)
							print("act_func in first layer: ", a1)
							print("dropout rate: ", d)
							print("optimizer: ", optimizer)
							print()
							
							

							train(epoch, lr, train_loader, val_loader, test_loader, model, optimizer)	


		print("Grid search on 3-Layered")	
		num_epochs = [25,50,75,100]
		learning_rates = [0.01,0.001,0.0005]
		batch_size = 64
		num_layer1 = [512, 256]
		num_layer2 = [256, 128]
		dropout = [0,0.2,0.8]
		act_func = ["ReLU","hardswish"]

		for lr in learning_rates:
			for n1 in num_layer1:
				for n2 in num_layer2:
					for f1 in act_func:
						for f2 in act_func:							
							for d in dropout:	
								for epoch in epochs:
									# Fix the randomness
									seed = 1234
									torch.manual_seed(seed)
									#(self, num_layer1, num_layer2, act_func1, act_func2, dropout)					
									model = ThreeLayer(n1,n2,a1,a2,d).to(device)
									optimizer = torch.optim.Adam(model.parameters(), lr=lr)							
							

									print()
									print("curr learning_rate: ", lr)
									print("num of neurons in the first layer: ", n1)
									print("num of neurons in the second layer: ", n2)
									print("act_func in first layer: ", a1)
									print("act_func in second layer: ", a2)
									print("dropout rate: ", d)
									print("optimizer: ", optimizer)
									print()
									
									

									train(epoch, lr, train_loader, val_loader, test_loader, model, optimizer)			





