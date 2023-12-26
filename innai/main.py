from torch.utils.data import DataLoader

from data_set.innai_data_set import InnAiDataSet

dataset = InnAiDataSet()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

for batch_idx, (precipitationMaps, currentInnLevels, nextInnLevels) in enumerate(dataloader):
    print(precipitationMaps.size())


# print(len(dataset))
# image, levels, prediction = dataset.__getitem__(5)

