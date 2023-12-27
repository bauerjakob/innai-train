using InnAiServer.Data.Collections;
using MongoDB.Bson;

namespace InnAiServer.Data.Repositories;

public interface IRainRadarRepository
{
    Task CreateAsync(RainRadar radarData);
    Task<ObjectId[]> GetLastIdsAsync(int count);
    Task<RainRadar> GetAsync(string id);
}