using InnAiServer.ApiClients;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;

namespace InnAiServer.Services;

public class RainRadarService : IRainRadarService
{
    private readonly ILogger<RainRadarService> _logger;
    private readonly IRainRadarClient _rainRadarClient;
    private readonly IRainRadarRepository _dbRepository;

    public RainRadarService(ILogger<RainRadarService> logger, IRainRadarClient rainRadarClient, IRainRadarRepository dbRepository)
    {
        _logger = logger;
        _rainRadarClient = rainRadarClient;
        _dbRepository = dbRepository;
    }
    
    public Task<RainRadar[]> GetLastAsync(int count)
    {
        return _dbRepository.GetLastAsync(count);
    }

    public async Task<byte[]> GetRadarImageAsync(string radarId)
    {
        var radarItem = await _dbRepository.GetAsync(radarId);
        return radarItem.Image;
    }

    public async Task DownloadAndStoreLatestRadarImagesAsync()
    {
        var items = await DownloadLatestRadarImagesFromApiAsync();
        
        foreach (var rainData in items)
        {
            await _dbRepository.CreateAsync(rainData);
        }
        
        _logger.LogInformation("[{ServiceName}] Successfully downloaded and stored {Count} items", nameof(RainRadarService), items?.Count() ?? 0);
    }
    
    private async Task<IEnumerable<RainRadar>> DownloadLatestRadarImagesFromApiAsync()
    {
        var latestItem = (await _dbRepository.GetLastAsync(1)).SingleOrDefault();

        var latestTime = latestItem?.Timestamp ?? DateTime.MinValue;
        
        var data = await _rainRadarClient.GetLatestRadarImageAsync(latestTime);

        return data
            .Select(x =>
                new RainRadar(x.Timestamp, x.Data));
    }
    
    private async Task<RainRadar?> GetLastAsync()
    {
        var latestItem = (await GetLastAsync(1)).SingleOrDefault();
        return latestItem;
    }
}