using InnAiServer.Dtos;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;

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
}