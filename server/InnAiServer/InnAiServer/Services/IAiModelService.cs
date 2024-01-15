using InnAiServer.Dtos;
using InnAiServer.Models;

namespace InnAiServer.Services;

public interface IAiModelService
{
    public Task<TrainingDataDto> GetTrainingDataAsync(int count, int predictHours);
}