using System.Numerics;
using InnAiServer.Data.Collections;
using MongoDB.Driver;

namespace InnAiServer.Data.Repositories;

public interface IInnLevelRepository
{
    public Task CreateAsync(InnLevel innLevel);
    public Task<InnLevel[]> GetLastAsync(int count);
}