using InnAiServer.ApiClients;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;
using InnAiServer.Options;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.Extensions.Options;

namespace InnAiServer.Services;

public class InnLevelService : IInnLevelService
{
    private readonly ILogger<InnLevelService> _logger;
    private readonly InnLevelOptions _options;
    private readonly IInnLevelClient _innLevelClient;
    private readonly IInnLevelRepository _innLevelRepository;

    public InnLevelService(ILogger<InnLevelService> logger, IOptions<InnLevelOptions> options, IInnLevelClient innLevelClient, IInnLevelRepository innLevelRepository)
    {
        _logger = logger;
        _options = options.Value;
        _innLevelClient = innLevelClient;
        _innLevelRepository = innLevelRepository;
    }

    public Task<InnLevel[]> GetLastAsync(string station, int count)
    {
        return _innLevelRepository.GetLastAsync(station, count);
    }
    
    public Task<InnLevel[]> GetLastAsync(string station, int count, DateTime before)
    {
        return _innLevelRepository.GetLastAsync(station, count, before);
    }

    public async Task DownloadAndStoreAsync()
    {
        foreach (var station in _options.Stations)
        {
            await DownloadAndStoreAsync(station);
        }
    }

    private async Task DownloadAndStoreAsync(InnStation station)
    {
        var lastItem = (await _innLevelRepository.GetLastAsync(station.Name, 1)).SingleOrDefault();
        var from = lastItem?.Timestamp.AddMinutes(1) ?? DateTime.Now.Subtract(TimeSpan.FromHours(5));
            
        var items = await _innLevelClient.GetLatestInnLevelsAsync(station, from);
            
        if (items != null)
        {
            foreach (var item in items)
            {
                await _innLevelRepository.CreateAsync(item);
            }
        }
        
        _logger.LogInformation("[{ServiceName}] Successfully downloaded and stored {Count} items - Station: {StationName}", nameof(InnLevelService), items?.Count() ?? 0, station.Name);
    }
}