import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler

file_path = "C:\\Users\\agozacan\\OneDrive - Humberside Fire and Rescue Service\\RBIP Project\\Merged Data\\clean_data.csv"

device = "cpu"
# device = "cuda"

df = pd.read_csv(file_path)
nrows = len(df)
ncols = len(df.columns)
noutput = len(df.iloc[:, -1].unique())

nEPOCHS = 256
BATCHSIZE = 16

print(f"There are {nrows} entries in the data")
print(f"There are {ncols-1} features")
print(f"There are {noutput} unique output values")

train_indices = np.random.choice(range(nrows), size = int(0.67 * nrows), replace = False)
test_indices = [i for i in range(nrows) if i not in train_indices]

class trainData(Dataset):

    def __init__(self, df, train_indices, oversample=False):

        if oversample:

            oversamp = RandomOverSampler()

            X, y = oversamp.fit_resample(df.iloc[train_indices, :-1], df.iloc[train_indices, -1])

            X = X.to_numpy()
            y = y.to_numpy()

        else:

            X = df.iloc[train_indices, :-1].to_numpy()
            y = df.iloc[train_indices, -1].to_numpy()

        self.X_train = torch.tensor(X, dtype = torch.float32)
        self.y_train = torch.tensor(y, dtype = torch.long)

        self.real_positives = sum(self.y_train)

    def __len__(self):

        return len(self.y_train)

    def __getitem__(self, idx):

        return self.X_train[idx], self.y_train[idx]

class testData(Dataset):

    def __init__(self, df, test_indices):

        X = df.iloc[test_indices, :-1].to_numpy()
        y = df.iloc[test_indices, -1].to_numpy()

        self.X_test = torch.tensor(X, dtype = torch.float32)
        self.y_test = torch.tensor(y, dtype = torch.long)

        self.real_positives = sum(self.y_test)

    def __len__(self):

        return len(self.y_test)

    def __getitem__(self, idx):
        
        return self.X_test[idx], self.y_test[idx]    

traindata = trainData(df, train_indices, oversample=True)
testdata = testData(df, test_indices)

print(f"There are {traindata.__len__()} entries in the training data, of which {traindata.real_positives} are real positives")
print(f"There are {testdata.__len__()} entries in the test data, of which {testdata.real_positives} are real positives")

trainset = torch.utils.data.DataLoader(traindata, batch_size = BATCHSIZE)
testset = torch.utils.data.DataLoader(testdata, batch_size = BATCHSIZE)

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(ncols-1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, noutput)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net().to(device)

# print(net)

class Evaluation():

    def __init__(self, testset, ncols):
        
        self.testset = testset
        self.ncols = ncols

    def print_acc(self):

        correct = 0
        total = 0

        with torch.no_grad():

            # for batch, (X, y) in enumerate(tqdm(self.testset)):
            for batch, (X, y) in enumerate(self.testset):

                X, y = X.to(device), y.to(device)
                output = net(X)

                for idx, x in enumerate(output):

                    if torch.argmax(x) == y[idx]:

                        correct += 1

                    total += 1

        print("Accuracy: ", round(correct/total, 3))

    def print_precision(self):

        true_positives = 0
        positives = 0

        with torch.no_grad():

            # for batch, (X, y) in enumerate(tqdm(self.testset)):
            for batch, (X, y) in enumerate(self.testset):

                X, y = X.to(device), y.to(device)
                output = net(X)

                for idx, x in enumerate(output):

                    if torch.argmax(x) == 1.0:
                        
                        positives += 1
                        
                        if torch.argmax(x) == y[idx]:

                            true_positives += 1

        if positives == 0:
            
            print("The algorithm assigned only zeros to the data!")

        else:

            print("Precision: ", round(true_positives/positives, 3))
        
        print(f"There were {positives} positives assigned altogether")

    def print_recall(self):

        true_positives = 0
        real_positives = 0

        with torch.no_grad():

            # for batch, (X, y) in enumerate(tqdm(self.testset)):
            for batch, (X, y) in enumerate(self.testset):

                X, y = X.to(device), y.to(device)
                output = net(X)

                for idx, x in enumerate(output):

                    if y[idx] == 1.0:
                        
                        real_positives += 1
                        
                        if torch.argmax(x) == y[idx]:

                            true_positives += 1

        print("Recall: ", round(true_positives/real_positives, 3))

eval = Evaluation(testset, ncols)

print("---RESULTS BEFORE TRAINING---")

eval.print_acc()
eval.print_precision()
eval.print_recall()

loss_function = nn.CrossEntropyLoss()

optimiser = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(nEPOCHS):

    # for batch, (X, y) in enumerate(tqdm(trainset)):
    for batch, (X, y) in enumerate(trainset):

        X, y = X.to(device), y.to(device)
        
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimiser.step()

    if epoch % 10 == 0:

        print(epoch, loss)

print("---RESULTS AFTER TRAINING---")

eval.print_acc()
eval.print_precision()
eval.print_recall()