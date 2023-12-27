using InnAiServer.Data.Collections;
using MongoDB.Bson;
using MongoDB.Driver;

namespace InnAiServer.Data.Repositories;

public class RainRadarRepository : IRainRadarRepository
{
    private readonly IMongoCollection<RainRadar> _rainRadarCollection;
    
    public RainRadarRepository(IMongoDatabase database)
    {
        var collection = database.GetCollection<RainRadar>(nameof(RainRadar));
        _rainRadarCollection = collection;
    }
    
    public async Task CreateAsync(RainRadar radarData)
    {
        await _rainRadarCollection.InsertOneAsync(radarData);
    }

    public Task<ObjectId[]> GetLastIdsAsync(int count)
    {
        var result = _rainRadarCollection.AsQueryable()
            .OrderByDescending(x => x.Timestamp)
            .Take(count)
            .Select(x => x.Id)
            .ToArray();
        
        return Task.FromResult(result);
    }

    public Task<RainRadar> GetAsync(string id)
    {
        var result = _rainRadarCollection
            .AsQueryable()
            .Single(x => x.Id == ObjectId.Parse(id));

        return Task.FromResult(result);
    }
}