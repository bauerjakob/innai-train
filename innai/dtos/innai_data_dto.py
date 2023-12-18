from typing_extensions import Self, List


class InnLevelDto:
    level: int
    station: str

    @staticmethod
    def from_json(json: any) -> Self:
        ret = InnLevelDto()
        ret.station = json["Station"]
        ret.level = json["Level"]
        return ret

class NextInnLevelDto:
    level: int
    hoursOffset: int

    @staticmethod
    def from_json(json: any) -> Self:
        ret = NextInnLevelDto()
        ret.level = json["Level"]
        ret.hoursOffset = json["HoursOffset"]
        return ret


class InnAiItemDto:
    timestamp: str
    inn_levels: List[InnLevelDto]
    precipitation_map: List[List[int]]
    next_inn_levels: List[NextInnLevelDto]

    @staticmethod
    def from_json(json: any) -> Self:
        ret = InnAiItemDto()
        ret.timestamp = json["Timestamp"]
        ret.inn_levels = [InnLevelDto.from_json(x) for x in json["InnLevels"]]
        ret.precipitation_map = json["PrecipitationMapValues"]
        ret.next_inn_levels = [NextInnLevelDto.from_json(x) for x in json["NextInnLevels"]]
        return ret

class InnAiDto:
    count: int
    items: List[InnAiItemDto]

    @staticmethod
    def from_json(json: any) -> Self:
        ret = InnAiDto()
        ret.count = json["Count"]
        ret.items = [InnAiItemDto.from_json(x) for x in json["Items"]]
        return ret


