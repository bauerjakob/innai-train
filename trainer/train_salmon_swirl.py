from data_set.innai_data_set import InnAiDataSet
from model.inn_ai_model import InnAiModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from model.salmon_swirl_model import SalmonSwirlModel
from trainer import Trainer

model = SalmonSwirlModel()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-4, lr_decay=0.00001)
loss_function = nn.MSELoss()
epochs = 100

trainer = Trainer(model, 32,8, loss_function, optimizer)

trainer.train(epochs)

print(trainer.evaluate_model())

f1 = plt.figure(1)
plt.plot(trainer.loss_history, 'b-')

f2 = plt.figure(2)
plt.plot([i[0] for i in trainer.model_performance_history], 'r-')

f3 = plt.figure(3)
plt.plot([i[1] for i in trainer.model_performance_history], 'g-')
plt.show()