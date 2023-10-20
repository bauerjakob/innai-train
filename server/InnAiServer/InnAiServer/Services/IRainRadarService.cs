using InnAiServer.Data.Collections;

namespace InnAiServer.Services;

public interface IRainRadarService
{
    public Task<RainRadar?> GetLatestEntryAsync();
    public Task<IEnumerable<RainRadar>> DownloadLatestRadarImagesFromApiAsync();

    public Task DownloadAndStoreLatestRadarImagesAsync();
}