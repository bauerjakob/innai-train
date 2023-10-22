using InnAiServer.Data.Collections;

namespace InnAiServer.Services;

public interface IRainRadarService
{
    public Task<RainRadar[]> GetLastAsync(int count);
    public Task<byte[]> GetRadarImageAsync(string radarId);
    public Task DownloadAndStoreLatestRadarImagesAsync();
}