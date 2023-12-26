using System.Text.Json;
using InnAiServer.Converters;
using InnAiServer.Dtos;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Bson.IO;

namespace InnAiServer.Controllers;

[ApiController]
[Route("api/v1/file")]
[ApiVersion("1.0")]
public class InnAiController : ControllerBase
{
    private readonly ILogger<InnAiController> _logger;
    private readonly IInnAiService _innAiService;

    public InnAiController(ILogger<InnAiController> logger, IInnAiService innAiService)
    {
        _logger = logger;
        _innAiService = innAiService;
    }

    [HttpPost("predict")]
    public async Task<ActionResult<PredictionResultDto>> PredictAsync([FromBody] PredictionInputDto dto)
    {
        await Task.CompletedTask;
        
        return Ok();
    }
}