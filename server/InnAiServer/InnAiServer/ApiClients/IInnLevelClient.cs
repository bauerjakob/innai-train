using InnAiServer.Data.Collections;
using InnAiServer.Dtos;

namespace InnAiServer.ApiClients;

public interface IInnLevelClient
{
    public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(InnStation station, DateTimeOffset from);
    
    public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(InnStation station, DateTimeOffset from, DateTimeOffset to);
}