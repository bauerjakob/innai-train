from InnAiClient.Dtos.InnLevelDto import InnLevelDto


class InnAiDataItemDto:
    Timestamp: str
    InnLevels: list[InnLevelDto]
    ImageId: str

    @staticmethod
    def FromJson(json):
        ret = InnAiDataItemDto()
        ret.Timestamp = json["timestamp"]
        ret.InnLevels = [InnLevelDto.FromJson(x) for x in json["innLevels"]]
        ret.ImageId = json["imageId"]
        return ret

