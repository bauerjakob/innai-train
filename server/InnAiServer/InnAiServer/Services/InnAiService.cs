using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Options;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Mvc.Razor.Internal;
using Microsoft.Extensions.Options;

namespace InnAiServer.Services;

public class InnAiService : IInnAiService
{
    private readonly ILogger<InnAiService> _logger;
    private readonly InnLevelOptions _innLevelOptions;
    private readonly IInnLevelService _innLevelService;
    private readonly IRainRadarService _rainRadarService;

    public InnAiService(ILogger<InnAiService> logger, IOptions<InnLevelOptions> innLevelOptions, IInnLevelService innLevelService, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _innLevelOptions = innLevelOptions.Value;
        _innLevelService = innLevelService;
        _rainRadarService = rainRadarService;
    }

    public async Task<InnAiDataDto> GetLastAsync(int count)
    {
        // var waterLevels = await _innLevelService.GetLastAsync(count * 10);
        var rainRadars = await _rainRadarService.GetLastAsync(count);

        // var items = MatchData(rainRadars, waterLevels);

        var stations = _innLevelOptions.Stations.Select(x => x.Name);

        List<InnAiDataItemDto> items = new ();
        foreach (var rainRadar in rainRadars)
        {
            List<InnLevelDto> innLevels = new();
            foreach (var station in stations)
            {
                var innLevel = await GetMatchingWaterLevelAsync(station, rainRadar.Timestamp);
                innLevels.Add(new InnLevelDto(innLevel?.Value, station));
            }

            var dateItem = new InnAiDataItemDto(rainRadar.Timestamp, innLevels.ToArray(), rainRadar.Id.ToString());
            items.Add(dateItem);
        }

        return new InnAiDataDto(items.Count, items.ToArray());
    }

    // private static InnAiDataItemDto[] MatchData(RainRadar[] rainRadars, InnLevel[] innLevels)
    // {
    //     List<InnAiDataItemDto> result = new();
    //
    //     var validRainRadars = rainRadars.OrderByDescending(x => x.Timestamp);
    //     var validInnLevels = innLevels.OrderBy(x => x.Timestamp).ToList();
    //
    //     foreach (var rainRadar in validRainRadars)
    //     {
    //         var timestamp = rainRadar.Timestamp;
    //
    //         InnLevel? innLevelMatch = null;
    //         InnLevel? previousInnLevel = null;
    //         foreach (var innLevel  in validInnLevels)
    //         {
    //             if (innLevel.Timestamp >= timestamp)
    //             {
    //                 var match = innLevel.Timestamp - timestamp;
    //                 var previousMatch = previousInnLevel?.Timestamp - timestamp;
    //
    //                 if (previousMatch.HasValue &&
    //                     Math.Abs(match.TotalMinutes) > 60 &&
    //                     Math.Abs(previousMatch.Value.TotalMinutes) > 60)
    //                 {
    //                     break;
    //                 }
    //                 
    //                 if (previousMatch.HasValue && previousMatch < match)
    //                 {
    //                     innLevelMatch = previousInnLevel;
    //                 }
    //                 else
    //                 {
    //                     innLevelMatch = innLevel;
    //                 }
    //             }
    //             
    //             previousInnLevel = innLevel;
    //         }
    //
    //         if (innLevelMatch is null && previousInnLevel is not null)
    //         {
    //             var diff = timestamp - previousInnLevel.Timestamp;
    //             if (diff < TimeSpan.FromMinutes(60))
    //             {
    //                 innLevelMatch = previousInnLevel;
    //             }
    //         }
    //
    //         if (innLevelMatch is not null)
    //         {
    //             result.Add(new InnAiDataItemDto(rainRadar.Timestamp, innLevelMatch.Value, rainRadar.Id.ToString() ?? throw new Exception()));
    //         }
    //     }
    //
    //     return result.ToArray();
    // }

    public async Task<InnLevel?> GetMatchingWaterLevelAsync(string station, DateTime timestamp)
    {
        var innLevels = await _innLevelService.GetLastAsync(station, 1, timestamp);
        var innLevel = innLevels.SingleOrDefault();
        
        if (innLevel is null)
        {
            return null;
        }

        var timeDifference = timestamp - innLevel.Timestamp;
        if (Math.Abs(timeDifference.TotalHours) > 3)
        {
            _logger.LogInformation("[{MethodName}] Found water level but it is too old - Station: {StationName}", nameof(GetMatchingWaterLevelAsync), station);
            return null;
        }

        return innLevel;
    } 
}