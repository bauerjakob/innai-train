import requests

from InnAiClient.Dtos.InnAiDataItemDto import InnAiDataItemDto


class InnAiClient:
    def __init__(self, baseUrl: str):
        self.baseUrl = baseUrl

    def GetData(self, count: int) -> list[InnAiDataItemDto]:
        response = requests.get(f"{self.baseUrl}/api/v1/data?count={count}")
        items = response.json()["items"]
        ret = []

        for item in items:
            ret.append(InnAiDataItemDto.FromJson(item))

        return ret
