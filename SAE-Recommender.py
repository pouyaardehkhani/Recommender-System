# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-1m/training_set.csv')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-1m/test_set.csv')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 25)
        self.fc5 = nn.Linear(25, 10)
        self.fc6 = nn.Linear(10, 25)
        self.fc7 = nn.Linear(25, 50)
        self.fc8 = nn.Linear(50, 100)
        self.fc9 = nn.Linear(100, 200)
        self.fc10 = nn.Linear(200, nb_movies)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.SiLU()
        self.activation4 = nn.Tanh()
    def forward(self, x):
        x = self.activation2(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.activation2(self.fc3(x))
        x = self.activation4(self.fc4(x))
        x = self.activation1(self.fc5(x))
        x = self.activation1(self.fc6(x))
        x = self.activation4(self.fc7(x))
        x = self.activation2(self.fc8(x))
        x = self.activation3(self.fc9(x))
        x = self.fc10(x)
        return x
    def train(self, criterion, optimizer, nb_epoch):
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(nb_users):
                input = Variable(training_set[id_user]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = self(input)
                    target.require_grad = False
                    output[target == 0] = 0
                    loss = criterion(output, target)
                    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data*mean_corrector)
                    s += 1.
                    optimizer.step()
            print('epoch: '+str(epoch)+' Train loss: '+str(train_loss/s))
    @staticmethod
    def test(model, criterion):
        # Testing the SAE
        test_loss = 0
        s = 0.
        for id_user in range(nb_users):
            input = Variable(training_set[id_user]).unsqueeze(0)
            target = Variable(test_set[id_user]).unsqueeze(0)
            if torch.sum(target.data > 0) > 0:
                output = model(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data*mean_corrector)
                s += 1.
        print('test loss: '+str(test_loss/s))
        


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(sae.parameters(), lr=0.01, weight_decay = 0.5)

# Training (error between real rating and predicted for example if loss==1 
# it means real rating and predicted rating will be different by 1 star 
# (our recommender system is 1 to 5 star rating per movie))
sae.train(criterion = criterion, optimizer = optimizer, nb_epoch=20)

# Test
SAE.test(sae,criterion = criterion)

