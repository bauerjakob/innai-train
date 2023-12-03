using InnAiServer.Dtos;
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
    public async Task<ActionResult<TrainingDataDto[]>> GetDataAsync([FromQuery] int count)
    {
        try
        {
            var result = await _aiModelService.GetTrainingDataAsync(count);
            return Ok(result);
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
    }
}