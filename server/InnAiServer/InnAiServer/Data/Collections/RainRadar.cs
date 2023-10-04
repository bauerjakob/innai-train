using MongoDB.Bson;

namespace InnAiServer.Data.Collections;

public abstract record RainRadar(ObjectId Id, DateTime Timestamp, byte[] Image);