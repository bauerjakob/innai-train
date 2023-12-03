using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record RainRadar(DateTime Timestamp, byte[] ImageRain, int[,] ValuesRain, int[,] ValuesRainReducedMin, int[,] ValuesRainReducedMax, int[,] ValuesRainReducedAvg)
{
    public ObjectId Id { get; set; }
}