using System.Numerics;
using InnAiServer.Data.Collections;
using MongoDB.Driver;

namespace InnAiServer.Data.Repositories;

public interface IInnLevelRepository
{
    public Task CreateAsync(InnLevel innLevel);
    public Task<InnLevel[]> GetLastAsync(string station, int count);
    public Task<InnLevel[]> GetLastAsync(string station, int count, DateTime dateTime);
}