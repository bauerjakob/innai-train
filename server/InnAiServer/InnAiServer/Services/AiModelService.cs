using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Options;
using Microsoft.Extensions.Options;

namespace InnAiServer.Services;

public class AiModelService : IAiModelService
{
    private readonly ILogger<AiModelService> _logger;
    private readonly IInnLevelService _innLevelService;
    private readonly IRainRadarService _rainRadarService;

    public AiModelService(ILogger<AiModelService> logger, IInnLevelService innLevelService, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _innLevelService = innLevelService;
        _rainRadarService = rainRadarService;
    }
    
    public async Task<InnAiDataDto> GetTrainingDataAsync(int count)
    {
        var rainRadars = await _rainRadarService.GetLastAsync(count);

        var stations = _innLevelService.GetInnStations().Select(x => x.Name);

        List<TrainingDataDto> items = new ();
        foreach (var rainRadar in rainRadars)
        {
            List<InnLevelDto> innLevels = new();
            foreach (var station in stations)
            {
                var innLevel = await GetMatchingWaterLevelAsync(station, rainRadar.Timestamp);
                innLevels.Add(new InnLevelDto(innLevel?.Value, station));
            }

            var dateItem = new TrainingDataDto(rainRadar.Timestamp, innLevels.ToArray(), rainRadar.Id.ToString());
            items.Add(dateItem);
        }

        return new InnAiDataDto(items.Count, items.ToArray());
    }
    
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