from typing import List

from torch.utils.data import Dataset
import json
import torch

from dtos.innai_data_dto import InnAiDto, InnAiItemDto


class InnAiDataSet(Dataset):
    def __init__(self):
        file = open('./data/data.json')
        data = json.load(file)

        innAiDto = InnAiDto.from_json(data)

        self.precipitationMaps = torch.tensor([item.precipitation_map for item in innAiDto.items])

        innLevels: List[int] = []
        nextInnLevels: List[int] = []
        for item in innAiDto.items:
            item: InnAiItemDto

            # current inn levels
            levels = [x.level for x in item.inn_levels[:3]]
            innLevels.append(levels)

            # next inn levels
            nextLevels = [x.level for x in item.next_inn_levels]
            nextInnLevels.append(nextLevels)

        self.innLevels = torch.tensor(innLevels)

        self.nextInnLevels = torch.tensor(nextInnLevels)




    def __getitem__(self, index):
        return self.precipitationMaps[index], self.innLevels[index], self.nextInnLevels[index]

    def __len__(self):
        return len(self.precipitationMaps)

