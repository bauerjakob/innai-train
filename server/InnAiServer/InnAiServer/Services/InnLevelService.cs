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

    public InnStation[] GetInnStations()
    {
        return _options.Stations;
    }

    // public async Task LoadMonthAsync(int year, int month)
    // {
    //     List<InnLevel> innLevels = new();
    //     foreach (var station in _options.Stations)
    //     {
    //         var stationData = await _innLevelClient.GetInnLevelsFromMonthAsync(station, year, month);    
    //         innLevels.AddRange(stationData);
    //     }
    //
    //     foreach (var item in innLevels)
    //     {
    //         await _innLevelRepository.CreateAsync(item);
    //     }
    // }
    
    public async Task LoadAsync(DateTime from)
    {
        List<InnLevel> innLevels = new();
        foreach (var station in _options.Stations)
        {
            var data = await _innLevelClient.GetInnLevelsAsync(station, from);
            innLevels.AddRange(data);
        }

        foreach (var innLevel in innLevels)
        {
            await _innLevelRepository.CreateAsync(innLevel);
        }
    }
}