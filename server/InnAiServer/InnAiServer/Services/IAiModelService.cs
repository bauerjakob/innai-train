using InnAi.Core;
using InnAiServer.Dtos;
using InnAiServer.Models;

namespace InnAiServer.Services;

public interface IAiModelService
{
    public Task<TrainingData> GetTrainingDataAsync(int count, int predictHours);
}