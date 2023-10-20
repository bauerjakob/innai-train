using InnAiServer.Data.Collections;
using InnAiServer.Dtos;

namespace InnAiServer.ApiClients;

public interface IInnLevelClient
{
    public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(DateTimeOffset from);
    
    public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(DateTimeOffset from, DateTimeOffset to);
}