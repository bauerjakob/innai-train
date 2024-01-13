from data_set.innai_data_set import InnAiDataSet
from model.inn_ai_model import InnAiModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from model.inn_ai_model import InnAiModel
from trainer import Trainer

model = InnAiModel()
optimizer = optim.Adam(model.parameters(), lr=0.00002)
loss_function = nn.MSELoss()
epochs = 50

trainer = Trainer(model, 10, 8, loss_function, optimizer)

trainer.train(epochs)

print(trainer.evaluate_model())  #output (0.0007225553085419677, tensor(12.7865))

f1 = plt.figure(1)
plt.plot(trainer.loss_history, 'b-')

f2 = plt.figure(2)
plt.plot([i[0] for i in trainer.model_performance_history], 'r-')

f3 = plt.figure(3)
plt.plot([i[1] for i in trainer.model_performance_history], 'g-')
plt.show()