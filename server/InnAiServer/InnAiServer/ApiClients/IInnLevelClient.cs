using InnAiServer.Data.Collections;
using InnAiServer.Dtos;

namespace InnAiServer.ApiClients;

public interface IInnLevelClient
{
    // public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(InnStation station, DateTimeOffset from);
    // public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(InnStation station, DateTimeOffset from, DateTimeOffset to);
    // public Task<IEnumerable<InnLevel>> GetInnLevelsFromMonthAsync(InnStation station, int year, int month);
    public Task<IEnumerable<InnLevel>> GetInnLevelsAsync(InnStation station, DateTimeOffset from);
}