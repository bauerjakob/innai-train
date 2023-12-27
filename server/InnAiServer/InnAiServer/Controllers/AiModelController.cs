using System.Text.Json;
using InnAi.Core;
using InnAiServer.Converters;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;
using InnAiServer.Dtos;
using InnAiServer.Models;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;

namespace InnAiServer.Controllers;

[Route("api/v1/aiModel")]
public class AiModelController : ControllerBase
{
    private readonly ILogger<AiModelController> _logger;
    private readonly IAiModelService _aiModelService;
    private readonly IFileRepository _fileRepository;

    public AiModelController(ILogger<AiModelController> logger, IAiModelService aiModelService, IFileRepository fileRepository)
    {
        _logger = logger;
        _aiModelService = aiModelService;
        _fileRepository = fileRepository;
    }

    [HttpGet("trainingData")]
    public async Task<ActionResult<FileResultDto>> GetDataAsync([FromQuery] int count,
        int predictHours)
    {
        var fileId = Guid.NewGuid();
        
        _ = Task.Run(async () =>
        {
            try
            {
                var result = await _aiModelService.GetTrainingDataAsync(count, predictHours);
                var options = new JsonSerializerOptions();
                // options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());

                var samples = SampleData(result.Items);

                foreach (var sample in samples)
                {
                    var json = JsonSerializer.Serialize(sample, options);

                    var ms = new MemoryStream();
                    var sw = new StreamWriter(ms);
                    await sw.WriteAsync(json);
                    await sw.FlushAsync();
                    ms.Position = 0;

                    var data = ms.ToArray();

                    var fileData = new FileData(fileId, data);
                    await _fileRepository.CreateAsync(fileData);
                    _logger.LogInformation($"File with id {fileId} successfully stored");
                }
            }
            catch (Exception e)
            {
                _logger.LogError(e, string.Empty);
            }
        });

        return Ok(new FileResultDto(fileId));
    }

    private List<List<TrainingDataItem>> SampleData(TrainingDataItem[] data)
    {
        List<List<TrainingDataItem>> result = new();
        List<TrainingDataItem> sample = new();
        for (int i = 0; i < data.Length; i++)
        {
            if (i % 1000 == 0 && sample.Count != 0)
            {
                result.Add(sample);
                sample = new();
            }
            
            sample.Add(data[i]);
        }

        if (sample.Count > 0)
        {
            result.Add(sample);
        }

        return result;
    }
}