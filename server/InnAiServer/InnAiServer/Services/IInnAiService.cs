using InnAiServer.Dtos;

namespace InnAiServer.Services;

public interface IInnAiService
{
    public Task<InnAiDataDto> GetLastAsync(int count);
}