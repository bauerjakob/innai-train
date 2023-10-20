using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record InnLevel(DateTime Timestamp, int Value)
{
    public ObjectId Id { get; set; }
}