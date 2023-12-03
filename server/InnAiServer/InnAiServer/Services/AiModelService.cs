using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Models;
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
    
    public async Task<TrainingDataDto> GetTrainingDataAsync(int count, PrecipitationValueMode mode, int predictHours)
    {
        var rainRadars = await _rainRadarService.GetLastAsync(count);

        var stations = _innLevelService.GetInnStations().Select(x => x.Name);

        List<TrainingDataItemDto> items = new ();
        
        foreach (var rainRadar in rainRadars)
        {
            List<InnLevelDto> innLevels = new();
            List<NextInnLevelDto> nextInnLevelDtos = new();
            foreach (var station in stations)
            {
                var innLevel = await GetMatchingWaterLevelAsync(station, rainRadar.Timestamp, predictHours);
                innLevels.Add(new InnLevelDto(innLevel.CurrentLevel.Value, station));

                if (station == "RosenheimAboveMangfallmündung")
                {
                    nextInnLevelDtos = innLevel
                        .NextLevels
                        .Select(x => 
                            new NextInnLevelDto(x.Value, (int)(x.Timestamp - innLevel.CurrentLevel.Timestamp).TotalHours)).ToList();
                }
            }

            var data = mode switch
            {
                PrecipitationValueMode.Min => rainRadar.ValuesRainReducedMin,
                PrecipitationValueMode.Max => rainRadar.ValuesRainReducedMax,
                PrecipitationValueMode.Avg => rainRadar.ValuesRainReducedAvg,
                _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null)
            };

            List<int[]> parsedData = new();

            for (int i = 0; i < data.GetLength(0); i++)
            {
                List<int> row = new();
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    row.Add(data[i, j]);
                }
                
                parsedData.Add(row.ToArray());
            }
            

            var dateItem = new TrainingDataItemDto(rainRadar.Timestamp, innLevels.ToArray(), rainRadar.Id.ToString(), parsedData.ToArray(), nextInnLevelDtos.ToArray());
            items.Add(dateItem);
        }

        return new TrainingDataDto(items.Count, items.ToArray());
    }
    
    public async Task<(InnLevel CurrentLevel, InnLevel[] NextLevels)> GetMatchingWaterLevelAsync(string station, DateTime timestamp, int predictHours)
    {
        var innLevels = (await _innLevelService.GetLastAsync(station, predictHours + 1, timestamp.AddHours(predictHours))).OrderBy(x => x.Timestamp);
        
        if (innLevels == null || innLevels.Count() != predictHours + 1)
        {
            throw new Exception();
        }
        
        var innLevel = innLevels.First();

        var nextInnLevels = innLevels.TakeLast(predictHours).ToArray();
        
        var timeDifference = timestamp - innLevel.Timestamp;
        if (Math.Abs(timeDifference.TotalHours) != 0)
        {
            _logger.LogInformation("[{MethodName}] Found water level but it is too old - Station: {StationName}", nameof(GetMatchingWaterLevelAsync), station);
            throw new Exception();
        }

        return (innLevel, nextInnLevels);
    } 
    
}