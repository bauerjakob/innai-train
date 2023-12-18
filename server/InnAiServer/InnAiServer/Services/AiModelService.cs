using InnAi.Core;
using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Models;
using InnAiServer.Options;
using Microsoft.Extensions.Options;
using InnLevel = InnAiServer.Data.Collections.InnLevel;

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
    
    public async Task<TrainingData> GetTrainingDataAsync(int count, PrecipitationValueMode mode, int predictHours)
    {
        var rainRadars = await _rainRadarService.GetLastAsync(count);

        var stations = _innLevelService.GetInnStations().Select(x => x.Name);

        List<TrainingDataItem> items = new ();
        
        foreach (var rainRadar in rainRadars)
        {
            List<InnAi.Core.InnLevel> innLevels = new();
            List<NextInnLevel> nextInnLevelDtos = new();
            try
            {
                foreach (var station in stations)
                {
                    var innLevel = await GetMatchingWaterLevelAsync(station, rainRadar.Timestamp, predictHours, station == "RosenheimAboveMangfallmündung");
                    innLevels.Add(new InnAi.Core.InnLevel(innLevel.CurrentLevel.Value, station));

                    if (station == "RosenheimAboveMangfallmündung")
                    {
                        nextInnLevelDtos = innLevel
                            .NextLevels
                            .Select(x => 
                                new NextInnLevel(x.Value, (int)(x.Timestamp - innLevel.CurrentLevel.Timestamp).TotalHours)).ToList();
                    }
                }
            }
            catch (Exception e)
            {
                _logger.LogError(e, string.Empty);
                continue;
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
            

            var dateItem = new TrainingDataItem(rainRadar.Timestamp, innLevels.ToArray(), rainRadar.Id.ToString(), parsedData.ToArray(), nextInnLevelDtos.ToArray());
            items.Add(dateItem);
        }

        return new TrainingData(items.Count, items.ToArray());
    }
    
    public async Task<(InnLevel CurrentLevel, InnLevel[] NextLevels)> GetMatchingWaterLevelAsync(string station, DateTime timestamp, int predictHours, bool loadNextLevels)
    {
        var count = (loadNextLevels ? predictHours : 0) + 1;
        var innLevels = (await _innLevelService.GetLastAsync(station, count, timestamp.AddHours(loadNextLevels ? predictHours : 0))).OrderBy(x => x.Timestamp);
        
        if (innLevels == null || innLevels.Count() != count)
        {
            throw new Exception();
        }
        
        var innLevel = innLevels.First();

        InnLevel[] nextInnLevels = loadNextLevels ? innLevels?.TakeLast(predictHours).ToArray() : null;
        
        var timeDifference = timestamp - innLevel.Timestamp;
        if (Math.Abs(timeDifference.TotalHours) != 0)
        {
            _logger.LogInformation("[{MethodName}] Found water level but it is too old - Station: {StationName}", nameof(GetMatchingWaterLevelAsync), station);
            throw new Exception();
        }

        return (innLevel, nextInnLevels);
    } 
    
}