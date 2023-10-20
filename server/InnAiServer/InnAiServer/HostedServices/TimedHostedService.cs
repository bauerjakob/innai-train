namespace InnAiServer.HostedServices;

public abstract class TimedHostedService<T> : IHostedService, IDisposable
{
    private readonly ILogger<TimedHostedService<T>> _logger;

    private readonly CancellationTokenSource _stopTokenSource = new CancellationTokenSource();

    protected TimedHostedService(ILogger<TimedHostedService<T>> logger)
    {
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Timed Hosted Service running.");

        var linkedToken = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _stopTokenSource.Token).Token;

        _ = Task.Run(async () =>
        {
            try
            {
                while (true)
                {
                    if (linkedToken.IsCancellationRequested) break;

                    try
                    {
                        var config = await WaitAsync(linkedToken);
                        if (linkedToken.IsCancellationRequested) break;
                        await DoWorkAsync(config, linkedToken);
                    }
                    catch (TaskCanceledException ex) when (ex.CancellationToken == linkedToken)
                    {
                        return;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, string.Empty);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, string.Empty);
                throw;
            }
            
            _logger.LogInformation("Timed Hosted Service stopped.");
        });
        return Task.CompletedTask;
    }

    public abstract Task DoWorkAsync(T config, CancellationToken cancellationToken);

    public abstract Task<T> WaitAsync(CancellationToken cancellationToken);

    public Task StopAsync(CancellationToken stoppingToken)
    {
        if (!_stopTokenSource.IsCancellationRequested)
        {
            _stopTokenSource.Cancel();
        }

        return Task.CompletedTask;
    }
    
    public void Dispose()
    {
        if (!_stopTokenSource.IsCancellationRequested)
        {
            _stopTokenSource.Cancel();
        }
        
        _stopTokenSource.Dispose();
    }
}