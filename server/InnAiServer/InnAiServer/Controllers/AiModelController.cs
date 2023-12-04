using System.Text.Json;
using InnAiServer.Converters;
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

    public AiModelController(ILogger<AiModelController> logger, IAiModelService aiModelService)
    {
        _logger = logger;
        _aiModelService = aiModelService;
    }
    
    [HttpGet("trainingData")]
    public async Task<IActionResult> GetDataAsync([FromQuery] int count, PrecipitationValueMode mode, int predictHours)
    {
        try
        {
            var result = await _aiModelService.GetTrainingDataAsync(count, mode, predictHours);
            
            var options = new JsonSerializerOptions();
            // options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());
            var json = JsonSerializer.Serialize(result, options);

            var ms = new MemoryStream();
            var sw = new StreamWriter(ms);
            await sw.WriteAsync(json);
            await sw.FlushAsync();
            ms.Position = 0;

            return File(ms, "application/json", $"{Guid.NewGuid()}.json");
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
    }
}