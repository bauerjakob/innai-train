using InnAiServer.Dtos;

namespace InnAiServer.Services;

public interface IAiModelService
{
    public Task<InnAiDataDto> GetTrainingDataAsync(int count);
}