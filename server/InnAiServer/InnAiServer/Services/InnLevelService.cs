using InnAiServer.ApiClients;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;

namespace InnAiServer.Services;

public class InnLevelService : IInnLevelService
{
    private readonly ILogger<InnLevelService> _logger;
    private readonly IInnLevelClient _innLevelClient;
    private readonly IInnLevelRepository _innLevelRepository;

    public InnLevelService(ILogger<InnLevelService> logger, IInnLevelClient innLevelClient, IInnLevelRepository innLevelRepository)
    {
        _logger = logger;
        _innLevelClient = innLevelClient;
        _innLevelRepository = innLevelRepository;
    }

    public async Task DownloadAndStoreLatestRadarImageAsync()
    {
        var lastItem = (await _innLevelRepository.GetLastAsync(1)).SingleOrDefault();
        var from = lastItem?.Timestamp ?? DateTime.Now.Subtract(TimeSpan.FromHours(5));
        
        var items = await _innLevelClient.GetLatestInnLevelsAsync(from);

        if (items != null)
        {
            foreach (var item in items)
            {
                await _innLevelRepository.CreateAsync(item);
            }
        }
    }
}