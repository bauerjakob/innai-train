using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;
using Microsoft.AspNetCore.Mvc;

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
        FileData file;
        try
        {
            file = await _fileRepository.GetAsync(fileId);
        }
        catch (Exception e)
        {
            _logger.LogError(e, string.Empty);
            return BadRequest();
        }

        var stream = new MemoryStream(file.Data);

        return File(stream, "application/json", $"{file.ExternalId.ToString().ToLower()}.json");
    }
}