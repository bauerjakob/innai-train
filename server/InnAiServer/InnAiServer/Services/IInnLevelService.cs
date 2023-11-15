using InnAiServer.Data.Collections;

namespace InnAiServer.Services;

public interface IInnLevelService
{
    public Task<InnLevel[]> GetLastAsync(string station, int count);
    public Task<InnLevel[]> GetLastAsync(string station, int count, DateTime before);
    public Task DownloadAndStoreAsync();
}