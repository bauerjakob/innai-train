using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record RainRadar(DateTime Timestamp, byte[] ImageRain, byte[] ImageRainSnow, int[,] DbzRain, int[,] DbzRainSnow)
{
    public ObjectId Id { get; set; }
}