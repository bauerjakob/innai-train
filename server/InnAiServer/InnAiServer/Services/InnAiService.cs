using InnAiServer.Data.Collections;
using InnAiServer.Dtos;

namespace InnAiServer.Services;

public class InnAiService : IInnAiService
{
    private readonly ILogger<InnAiService> _logger;
    private readonly IInnLevelService _innLevelService;
    private readonly IRainRadarService _rainRadarService;

    public InnAiService(ILogger<InnAiService> logger, IInnLevelService innLevelService, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _innLevelService = innLevelService;
        _rainRadarService = rainRadarService;
    }

    public async Task<InnAiDataDto> GetLastAsync(int count)
    {
        var waterLevels = await _innLevelService.GetLastAsync(count * 10);
        var rainRadars = await _rainRadarService.GetLastAsync(count);

        var items = MatchData(rainRadars, waterLevels);

        return new InnAiDataDto(items.Length, items);
    }

    private static InnAiDataItemDto[] MatchData(RainRadar[] rainRadars, InnLevel[] innLevels)
    {
        List<InnAiDataItemDto> result = new();

        var validRainRadars = rainRadars.OrderByDescending(x => x.Timestamp);
        var validInnLevels = innLevels.OrderBy(x => x.Timestamp).ToList();

        foreach (var rainRadar in validRainRadars)
        {
            var timestamp = rainRadar.Timestamp;

            InnLevel? innLevelMatch = null;
            InnLevel? previousInnLevel = null;
            foreach (var innLevel  in validInnLevels)
            {
                if (innLevel.Timestamp >= timestamp)
                {
                    var match = innLevel.Timestamp - timestamp;
                    var previousMatch = previousInnLevel?.Timestamp - timestamp;

                    if (previousMatch.HasValue &&
                        Math.Abs(match.TotalMinutes) > 60 &&
                        Math.Abs(previousMatch.Value.TotalMinutes) > 60)
                    {
                        break;
                    }
                    
                    if (previousMatch.HasValue && previousMatch < match)
                    {
                        innLevelMatch = previousInnLevel;
                    }
                    else
                    {
                        innLevelMatch = innLevel;
                    }
                }
                
                previousInnLevel = innLevel;
            }

            if (innLevelMatch is null && previousInnLevel is not null)
            {
                var diff = timestamp - previousInnLevel.Timestamp;
                if (diff < TimeSpan.FromMinutes(60))
                {
                    innLevelMatch = previousInnLevel;
                }
            }

            if (innLevelMatch is not null)
            {
                result.Add(new InnAiDataItemDto(rainRadar.Timestamp, innLevelMatch.Value, rainRadar.Id.ToString() ?? throw new Exception()));
            }
        }

        return result.ToArray();
    }
}