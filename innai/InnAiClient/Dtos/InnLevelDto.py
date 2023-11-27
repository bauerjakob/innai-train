class InnLevelDto:
    Level: int
    Station: str

    @staticmethod
    def FromJson(json):
        ret = InnLevelDto()
        ret.Station = json["station"]
        ret.Level = json["level"]
        return ret
