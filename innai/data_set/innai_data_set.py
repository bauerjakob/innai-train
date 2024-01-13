from typing import List

from torch.utils.data import Dataset
import json
import torch
import numpy as np

from dtos.innai_data_dto import InnAiDto, InnAiItemDto


def normalize(input, min: int, max: int) -> List[float]:
    out = []

    for item in input:
        out.append((item - min) / (max - min))

    return out


class InnAiDataSet(Dataset):
    def __init__(self, imageSize: int):
        file = open('./data/data.json')
        data = json.load(file)
        self.imageSize = imageSize

        innAiDto = InnAiDto.from_json(data)

        if imageSize == 8:
            self.precipitationMaps = torch.tensor([np.array(item.precipitation_map_small).flatten() for item in innAiDto.items], dtype=torch.float32)
        elif imageSize == 16:
            self.precipitationMaps = torch.tensor(
                [np.array(item.precipitation_map_medium).flatten() for item in innAiDto.items], dtype=torch.float32)
        elif imageSize == 32:
            self.precipitationMaps = torch.tensor(
                [np.array(item.precipitation_map_large).flatten() for item in innAiDto.items], dtype=torch.float32)



        innLevels: List[int] = []
        nextInnLevels: List[int] = []
        for item in innAiDto.items:
            item: InnAiItemDto

            # current inn levels
            levels = [float(x.level) for x in item.inn_levels[:3]]
            innLevels.append(levels)

            # next inn levels
            nextLevels = [float(x.level) for x in item.next_inn_levels[0:24]]
            nextInnLevels.append(nextLevels)

        self.innLevels = torch.tensor(innLevels)
        self.nextInnLevels = torch.tensor(nextInnLevels)




    def __getitem__(self, index):
        return (self.precipitationMaps[index],
                torch.tensor(normalize(self.innLevels[index], 0, 700)),
                torch.tensor(normalize(self.nextInnLevels[index], 0, 700)))

    def __len__(self):
        return len(self.precipitationMaps)

