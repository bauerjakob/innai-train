using InnAiServer.Data.Collections;

namespace InnAiServer.Data.Repositories;

public interface IFileRepository
{
    public Task CreateAsync(FileData fileData);
    public Task<FileData> GetAsync(Guid fileId);
}