using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record RainRadar(DateTime Timestamp, byte[] ImageRain, int[,] ValuesRain, double[,] ValuesRainLarge, double[,] ValuesRainMedium, double[,] ValuesRainSmall)
{
    public ObjectId Id { get; set; }
}