using System.Text.Json;
using System.Text.Json.Serialization;
using InnAi.Core;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Bson.IO;

namespace InnAiServer.Controllers;

[ApiController]
[Route("api/v1/innAi")]
[ApiVersion("1.0")]
public class FileController : ControllerBase
{
    private readonly ILogger<FileController> _logger;
    private readonly IFileRepository _fileRepository;

    public FileController(ILogger<FileController> logger, IFileRepository fileRepository)
    {
        _logger = logger;
        _fileRepository = fileRepository;
    }
    
    [HttpPost("{fileId}")]
    public async Task<IActionResult> GetAsync([FromRoute] Guid fileId)
    {
        FileData[] fileSamples;
        try
        {
            fileSamples = await _fileRepository.GetAsync(fileId);
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return BadRequest();
        }

        var ret = new List<TrainingDataItem>();


        foreach (var sample in fileSamples)
        {
            var stream = new MemoryStream(sample.Data);
            var data =  await JsonSerializer.DeserializeAsync<TrainingDataItem[]>(stream);
            ret.AddRange(data);
        }

        var trainingData = new TrainingData(ret.Count, ret.ToArray());

        var json = JsonSerializer.Serialize(trainingData);

        var ms = new MemoryStream();
        var sw = new StreamWriter(ms);
        await sw.WriteAsync(json);
        await sw.FlushAsync();
        ms.Position = 0;

        // var data = ms.ToArray();

        return File(ms, "application/json", $"{fileId.ToString().ToLower()}.json");
    }
}