// using InnAiServer.Services;
// using NCrontab;
//
// namespace InnAiServer.HostedServices;
//
// public record InnLevelHostedServicePram();
//
// public class InnLevelHostedService : TimedHostedService<InnLevelHostedServicePram>
// {
//     private readonly IServiceScopeFactory _scopeFactory;
//     private readonly CrontabSchedule _schedule = CrontabSchedule.Parse("* * * * *");
//
//     public InnLevelHostedService(ILogger<TimedHostedService<InnLevelHostedServicePram>> logger, IServiceScopeFactory scopeFactory) : base(logger)
//     {
//         _scopeFactory = scopeFactory;
//     }
//
//     public override async Task DoWorkAsync(InnLevelHostedServicePram config, CancellationToken cancellationToken)
//     {
//         using var scope = _scopeFactory.CreateScope();
//         var innLevelService = scope.ServiceProvider.GetRequiredService<IInnLevelService>();
//
//         await innLevelService.DownloadAndStoreAsync();            
//     }
//
//     public override async Task<InnLevelHostedServicePram> WaitAsync(CancellationToken cancellationToken)
//     {
//         var now = DateTime.UtcNow;
//         var nextOccurrence = _schedule.GetNextOccurrence(now);
//         await Task.Delay(nextOccurrence - now, cancellationToken);
//
//         return new InnLevelHostedServicePram();
//     }
// }