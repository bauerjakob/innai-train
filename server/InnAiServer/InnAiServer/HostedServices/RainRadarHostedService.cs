using InnAiServer.ApiClients;
using InnAiServer.Data.Repositories;
using InnAiServer.Services;
using NCrontab;

namespace InnAiServer.HostedServices;

public record RainRadarHostedServicePram();

public class RainRadarHostedService : TimedHostedService<RainRadarHostedServicePram>
{
    private readonly ILogger<TimedHostedService<RainRadarHostedServicePram>> _logger;
    private readonly IServiceScopeFactory _scopeFactory;

    private readonly CrontabSchedule _schedule = CrontabSchedule.Parse("* * * * *");

    public RainRadarHostedService(ILogger<TimedHostedService<RainRadarHostedServicePram>> logger, IServiceScopeFactory scopeFactory) : base(logger)
    {
        _logger = logger;
        _scopeFactory = scopeFactory;
    }

    public override async Task DoWorkAsync(RainRadarHostedServicePram config, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Timed Hosted Service is working.");
        
        using var scope = _scopeFactory.CreateScope();
        var rainRadarService = scope.ServiceProvider.GetRequiredService<IRainRadarService>();

        // await rainRadarService.DownloadAndStoreLatestRadarImagesAsync();
    }

    public override async Task<RainRadarHostedServicePram> WaitAsync(CancellationToken cancellationToken)
    {
        var now = DateTime.UtcNow;
        var nextOccurrence = _schedule.GetNextOccurrence(now);
        await Task.Delay(nextOccurrence - now, cancellationToken);

        return new RainRadarHostedServicePram();
    }
}