using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record RainRadar(DateTime Timestamp, byte[] Image)
{
    public ObjectId Id { get; set; }
}