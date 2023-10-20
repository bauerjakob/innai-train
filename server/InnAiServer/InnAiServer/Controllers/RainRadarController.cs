using InnAiServer.Dtos;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;

namespace InnAiServer.Controllers;

[ApiController]
[Route("api/v1")]
[ApiVersion("1.0")]
public class RainRadarController : ControllerBase
{
    private readonly ILogger<RainRadarController> _logger;
    private readonly IRainRadarService _rainRadarService;

    public RainRadarController(ILogger<RainRadarController> logger, IRainRadarService rainRadarService)
    {
        _logger = logger;
        _rainRadarService = rainRadarService;
    }

    [HttpGet("inn-levels")]
    public async Task<ActionResult<InnLevelDto[]>> GetInnLevels(int count)
    {
        await Task.CompletedTask;
        return Ok(null);
    }

    [HttpGet("radarImage/latest")]
    public async Task<IActionResult> GetLatestRadarImageAsync()
    {
        try
        {
            var latestEntry = await _rainRadarService.GetLatestEntryAsync();
            if (latestEntry == null)
            {
                _logger.LogError("Entry cannot be null", string.Empty);
                return StatusCode(StatusCodes.Status500InternalServerError);    
            }

            return File(latestEntry.Image, "image/png", "radar.png");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, string.Empty);
            return StatusCode(StatusCodes.Status500InternalServerError);
        }
    }
}