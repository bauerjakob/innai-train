using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record InnStation(string Name, string Id);

public record InnLevel(DateTime Timestamp, int Value, InnStation Station)
{
    public ObjectId Id { get; set; }
}