class InnAiDataItemDto:
    Timestamp: str
    Level: int
    ImageId: str

    @staticmethod
    def FromJson(json):
        ret = InnAiDataItemDto()
        ret.Timestamp = json["timestamp"]
        ret.Level = json["level"]
        ret.ImageId = json["imageId"]

        return ret


