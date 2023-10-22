using InnAiServer.Data.Collections;

namespace InnAiServer.Services;

public interface IInnLevelService
{
    public Task<InnLevel[]> GetLastAsync(int count);
    public Task DownloadAndStoreAsync();
}