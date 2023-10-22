using InnAiServer.Data.Collections;
using MongoDB.Bson;

namespace InnAiServer.Data.Repositories;

public interface IRainRadarRepository
{
    Task CreateAsync(RainRadar radarData);
    Task<RainRadar[]> GetLastAsync(int count);
    Task<RainRadar> GetAsync(string id);
}