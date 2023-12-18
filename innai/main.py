from InnAiClient.Dtos.InnAiDataItemDto import InnAiDataItemDto
from InnAiClient.InnAiClient import InnAiClient

for item in data:
    print(item.ImageId, item.Level, item.Timestamp)
