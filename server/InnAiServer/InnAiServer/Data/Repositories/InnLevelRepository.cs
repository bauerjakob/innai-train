using InnAiServer.Data.Collections;
using MongoDB.Driver;

namespace InnAiServer.Data.Repositories;

public class InnLevelRepository : IInnLevelRepository
{
    private readonly IMongoCollection<InnLevel> _innLevelCollection;

    public InnLevelRepository(IMongoDatabase database)
    {
        var collection = database.GetCollection<InnLevel>(nameof(InnLevel));
        _innLevelCollection = collection;
    }
    
    public async Task CreateAsync(InnLevel innLevel)
    {
        await _innLevelCollection.InsertOneAsync(innLevel);
    }

    public Task<InnLevel[]> GetLastAsync(int count)
    {
        var result = _innLevelCollection.AsQueryable()
            .OrderByDescending(x => x.Timestamp)
            .Take(count)
            .ToArray();
        
        return Task.FromResult(result);
    }
}