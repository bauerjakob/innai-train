using InnAiServer.ApiClients;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;

namespace InnAiServer.Services;

public class RainRadarService : IRainRadarService
{
    private readonly IRainRadarClient _rainRadarClient;
    private readonly IRainRadarRepository _dbRepository;

    public RainRadarService(IRainRadarClient rainRadarClient, IRainRadarRepository dbRepository)
    {
        _rainRadarClient = rainRadarClient;
        _dbRepository = dbRepository;
    }

    public async Task<RainRadar?> GetLatestEntryAsync()
    {
        var latestItem = (await _dbRepository.GetLastAsync(1)).SingleOrDefault();
        return latestItem;
    }
    
    public async Task<IEnumerable<RainRadar>> DownloadLatestRadarImagesFromApiAsync()
    {
        var latestItem = (await _dbRepository.GetLastAsync(1)).SingleOrDefault();

        var latestTime = latestItem?.Timestamp ?? DateTime.MinValue;
        
        var data = await _rainRadarClient.GetLatestRadarImageAsync(latestTime);

        return data
            .Select(x =>
                new RainRadar(x.Timestamp, x.Data));
    }
    
    public async Task DownloadAndStoreLatestRadarImagesAsync()
    {
        var result = await DownloadLatestRadarImagesFromApiAsync();
        foreach (var rainData in result)
        {
            await _dbRepository.CreateAsync(rainData);
        }
    }
}