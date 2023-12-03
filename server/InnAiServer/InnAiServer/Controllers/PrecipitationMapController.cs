using System.Text.Json;
using InnAiServer.Converters;
using InnAiServer.Models;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;

namespace InnAiServer.Controllers;

[Route("api/v1/precipitationMap")]
public class PrecipitationMapController : ControllerBase
{
    private readonly ILogger<PrecipitationMapController> _logger;
    private readonly IRainRadarService _rainRadarService;

    public PrecipitationMapController(ILogger<PrecipitationMapController> logger, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _rainRadarService = rainRadarService;
    }
    
    [HttpGet("load")]
    public async Task<IActionResult> LoadDataAsync([FromQuery] int year, [FromQuery] int month)
    {
        var now = DateTime.UtcNow;

        if (year > now.Year || year == now.Year && month >= now.Month)
        {
            return BadRequest();
        }
        
        try
        {
            await _rainRadarService.LoadMonthAsync(year, month);
            // await _innLevelService.LoadMonthAsync(year, month);
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return BadRequest();
        }

        return Ok();
    }
    
    [HttpGet("{contentId}")]
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
    
    [HttpGet("{contentId}/values")]
    public async Task<ActionResult<IEnumerable<IEnumerable<int>>>> GetLatestRadarImageDbzAsync([FromRoute] string contentId)
    {
        try
        {
            var result = await _rainRadarService.GetRadarImageValuesAsync(contentId);
            
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