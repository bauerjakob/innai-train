using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public record FileData(Guid ExternalId, byte[] Data)
{
    public ObjectId Id { get; set; }
}