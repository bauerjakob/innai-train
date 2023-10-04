using InnAiServer.Dtos;
using Microsoft.AspNetCore.Mvc;

namespace InnAiServer.Controllers;

[ApiController]
[Route("api/v1")]
[ApiVersion("1.0")]
public class DataController : ControllerBase
{
    private readonly ILogger<DataController> _logger;

    public DataController(ILogger<DataController> logger)
    {
        _logger = logger;
    }

    [HttpGet("inn-levels")]
    public async Task<ActionResult<InnLevel[]>> GetInnLevels(int count)
    {
        await Task.CompletedTask;
        return Ok(null);
    }
}