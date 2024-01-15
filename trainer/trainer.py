import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_set.innai_data_set import InnAiDataSet


class Trainer:
    def __init__(self, model: nn.Module, batch_size: int, dataset_size: int, loss_function, optimizer):
        self.load_dataset(batch_size, dataset_size)
        self.epoch = 0
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.loss_history = []
        self.model_performance_history = []

    def load_dataset(self, batch_size, dataset_size: int):
        full_dataset = InnAiDataSet(dataset_size)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=1)

    def run_epoch(self):
        self.epoch += 1

        self.model.train()

        for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(self.train_data_loader):
            # precipitationData = Variable(precipitationMaps)
            # currentInnLevelsData = Variable(currentInnLevels)

            input = torch.cat((precipitationMaps, currentInnLevels), 1)
            inputVariable = Variable(input)
            self.optimizer.zero_grad()

            out = self.model(inputVariable)

            loss = self.loss_function(out, nextInnLevels)
            self.loss_history.append(loss.data)

            loss.backward()

            self.optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(self.epoch, batch_idx * len(precipitationMaps), loss.data))

            torch.save(self.model.state_dict(), "checkpoints/model.pt")

    def calculate_original_loss(self, actualNormalized: torch.Tensor, predictionNormalized: torch.Tensor):
        min = 0
        max = 700

        actual = actualNormalized * (max - min) + min
        prediction = predictionNormalized * (max - min) + min

        diff = torch.abs(actual - prediction)

        return torch.mean(diff, dtype=torch.float).data

    def evaluate_model(self):
        losses = []
        original_losses = []

        for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(self.test_data_loader):
            input = torch.cat((precipitationMaps, currentInnLevels), 1)
            inputVariable = Variable(input)
            self.optimizer.zero_grad()

            out = self.model(inputVariable)

            loss = self.loss_function(out, nextInnLevels).item()
            losses.append(loss)

            original_loss = self.calculate_original_loss(out, nextInnLevels)

            original_losses.append(original_loss)

        return sum(losses) / len(losses), sum(original_losses) / len(original_losses)

    def train(self, epochs):
        for epoch in range(epochs):
            self.run_epoch()
            self.model_performance_history.append(self.evaluate_model())


