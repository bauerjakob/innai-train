using InnAiServer.Data.Collections;
using MongoDB.Driver;

namespace InnAiServer.Data.Repositories;

public class FileRepository : IFileRepository
{
    private readonly IMongoCollection<FileData> _fileCollection;

    public FileRepository(IMongoDatabase database)
    {
        var collection = database.GetCollection<FileData>(nameof(FileData));
        _fileCollection = collection;
    }
    
    public Task CreateAsync(FileData fileData)
    {
        return _fileCollection.InsertOneAsync(fileData);
    }

    public Task<FileData> GetAsync(Guid fileId)
    {
        var file = _fileCollection.AsQueryable().Where(x => x.ExternalId == fileId).Single() ?? throw new Exception();
        return Task.FromResult(file);
    }
}