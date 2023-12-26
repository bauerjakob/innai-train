from data_set.innai_data_set import InnAiDataSet
from model.inn_ai_model import InnAiModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch

dataset = InnAiDataSet()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


model = InnAiModel()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.MSELoss()
epochs = 3000
epoch = 0

def train(epoch):
    global model, optimizer, criterion

    model.train()

    for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(dataloader):
        precipitationData = Variable(precipitationMaps)
        currentInnLevelsData = Variable(currentInnLevels)

        optimizer.zero_grad()

        out = model(precipitationData, currentInnLevelsData)

        loss = criterion(out, nextInnLevels)

        loss.backward()

        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, batch_idx * len(precipitationMaps), loss.data))

        torch.save(model, "./checkpoints/innAiModel.pth")

while True:
    epoch += 1
    train(epoch)

for epoch in range(epochs):
    train(epoch)
