using InnAi.Core;
using InnAi.Model.Models;
using Microsoft.ML.Data;

namespace InnAi.Model;

public static class DataExtensions
{
    public static ModelInput ToModelInput(this TrainingDataItem data)
    {
        return new ModelInput
        {
            PrecipitationMap = data.PrecipitationMapValues.SelectMany(x => x).Select(x => (float)x).ToArray(),
            InnLevels = data.InnLevels.OrderBy(x => x.Station).Select(x => x.Level ?? throw new Exception()).Select(x => (float)x).ToArray(),
            Predictions = data.NextInnLevels.Select(x => (float)(x.Level ?? throw new Exception())).ToArray()
        };
    }
}