from data_set.innai_data_set import InnAiDataSet
from model.inn_ai_model import InnAiModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

full_dataset = InnAiDataSet()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1)

# dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

model = InnAiModel()
optimizer = optim.Adam(model.parameters(), lr=0.00002)
criterion = nn.MSELoss()
epochs = 10

loss_values = []
process_values = []

def evaluate_model():
    global model, criterion

    losses = []

    for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(test_dataset):
        input = torch.cat((precipitationMaps, currentInnLevels), 0)
        inputVariable = Variable(input)
        optimizer.zero_grad()

        out = model(inputVariable)

        loss = criterion(out, nextInnLevels)
        losses.append(loss)

    return (sum(losses) / len(losses)).item()

def train(epoch):
    global model, optimizer, criterion, loss_values

    if epoch >= 30:
        optimizer = optim.Adam(model.parameters(), lr=0.009)

    model.train()

    for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(train_data_loader):
        # precipitationData = Variable(precipitationMaps)
        # currentInnLevelsData = Variable(currentInnLevels)

        input = torch.cat((precipitationMaps, currentInnLevels), 1)
        inputVariable = Variable(input)
        optimizer.zero_grad()

        out = model(inputVariable)

        loss = criterion(out, nextInnLevels)
        loss_values.append(loss.data)

        loss.backward()

        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, batch_idx * len(precipitationMaps), loss.data))

        torch.save(model.state_dict(), "./checkpoints/model.pt")

    process_values.append(evaluate_model())

for epoch in range(epochs):
    train(epoch)

print(evaluate_model())

f1 = plt.figure(1)
plt.plot(loss_values, 'b-')

f2 = plt.figure(2)
plt.plot(process_values, 'r-')
plt.show()