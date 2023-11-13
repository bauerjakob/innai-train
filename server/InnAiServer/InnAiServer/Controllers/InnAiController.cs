using System.Text.Json;
using InnAiServer.Converters;
using InnAiServer.Dtos;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Bson.IO;

namespace InnAiServer.Controllers;

[ApiController]
[Route("api/v1")]
[ApiVersion("1.0")]
public class InnAiController : ControllerBase
{
    private readonly ILogger<InnAiController> _logger;
    private readonly IInnAiService _innAiService;
    private readonly IRainRadarService _rainRadarService;

    public InnAiController(ILogger<InnAiController> logger, IInnAiService innAiService, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _innAiService = innAiService;
        _rainRadarService = rainRadarService;
    }

    [HttpGet("data")]
    public async Task<ActionResult<InnAiDataItemDto[]>> GetDataAsync([FromQuery] int count)
    {
        try
        {
            var result = await _innAiService.GetLastAsync(count);
            return Ok(result);
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
        
    }

    [HttpGet("radarImage/{contentId}")]
    public async Task<IActionResult> GetLatestRadarImageAsync([FromRoute] string contentId)
    {
        try
        {
            var imageData = await _rainRadarService.GetRadarImageAsync(contentId);
            
            return File(imageData, "image/png", $"{contentId}.png");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
    }
    
    [HttpGet("radarImage/dbz/{contentId}")]
    public async Task<ActionResult<IEnumerable<IEnumerable<int>>>> GetLatestRadarImageDbzAsync([FromRoute] string contentId)
    {
        try
        {
            var result = await _rainRadarService.GetRadarImageDbzAsync(contentId);
            
            var options = new JsonSerializerOptions();
            options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());
            var json = JsonSerializer.Serialize(result, options);

            var ms = new MemoryStream();
            var sw = new StreamWriter(ms);
            await sw.WriteAsync(json);
            await sw.FlushAsync();
            ms.Position = 0;
            
            return File(ms, "application/json", $"{contentId}.json");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
    }
}