using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Options;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Mvc.Razor.Internal;
using Microsoft.Extensions.Options;

namespace InnAiServer.Services;

public class InnAiService : IInnAiService
{
    private readonly ILogger<InnAiService> _logger;
    private readonly IAiModelService _aiModelService;
    private readonly InnLevelOptions _innLevelOptions;

    public InnAiService(ILogger<InnAiService> logger, IAiModelService aiModelService)
    {
        _logger = logger;
        _aiModelService = aiModelService;
    }
    
}