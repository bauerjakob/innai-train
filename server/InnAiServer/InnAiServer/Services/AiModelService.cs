using System.Collections.Concurrent;
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
    
    public async Task<TrainingData> GetTrainingDataAsync(int count, int predictHours)
    {
        var rainRadarIds = await _rainRadarService.GetLastIdAsync(count);

        var stations = _innLevelService.GetInnStations().Select(x => x.Name);

        ConcurrentBag<TrainingDataItem> items = new ();

        int index = 0;

        await Parallel.ForEachAsync(rainRadarIds, async (id, token) =>
        {
            var rainRadar = await _rainRadarService.GetAsync(id);

            _logger.LogInformation("GetTrainingDataAsync - {0}/{1}", ++index, rainRadarIds.Length);
            List<InnAi.Core.InnLevel> innLevels = new();
            List<NextInnLevel> nextInnLevelDtos = new();
            try
            {
                foreach (var station in stations)
                {
                    var innLevel = await GetMatchingWaterLevelAsync(station, rainRadar.Timestamp, predictHours, station == "RosenheimAboveMangfall");
                    innLevels.Add(new InnAi.Core.InnLevel(innLevel.CurrentLevel.Value, station));

                    if (station == "RosenheimAboveMangfall")
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
                return;
            }


            var dataLarge = ParseData(rainRadar.ValuesRainLarge);
            var dataMedium = ParseData(rainRadar.ValuesRainMedium);
            var dataSmall = ParseData(rainRadar.ValuesRainSmall);

            var dateItem = new TrainingDataItem(rainRadar.Timestamp, innLevels.ToArray(), rainRadar.Id.ToString(), dataLarge, dataMedium, dataSmall, nextInnLevelDtos.ToArray());
            items.Add(dateItem);
        });
        
        return new TrainingData(items.Count, items.ToArray());
    }

    private double[][] ParseData(double[,] data)
    {
        List<double[]> parsedData = new();

        for (int i = 0; i < data.GetLength(0); i++)
        {
            List<double> row = new();
            for (int j = 0; j < data.GetLength(1); j++)
            {
                row.Add(data[i, j]);
            }
                
            parsedData.Add(row.ToArray());
        }

        return parsedData.ToArray();
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