using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc;

namespace InnAiServer.Controllers;

[Route("api/v1/innLevel")]
public class InnLevelController : ControllerBase
{
    private readonly ILogger<InnLevelController> _logger;
    private readonly IInnLevelService _innLevelService;

    public InnLevelController(ILogger<InnLevelController> logger, IInnLevelService innLevelService)
    {
        _logger = logger;
        _innLevelService = innLevelService;
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
            await _innLevelService.LoadAsync(new DateTime(year, month, 1));
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return BadRequest();
        }

        return Ok();
    }
}