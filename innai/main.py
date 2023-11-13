from InnAiClient.Dtos.InnAiDataItemDto import InnAiDataItemDto
from InnAiClient.InnAiClient import InnAiClient


def LoadData() -> list[InnAiDataItemDto]:
    client = InnAiClient("http://localhost:5294")
    data = client.GetData(50)
    return data


if __name__ == '__main__':
    data = LoadData()

    for item in data:
        print(item.ImageId, item.Level, item.Timestamp)
