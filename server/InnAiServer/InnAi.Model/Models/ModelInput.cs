using Microsoft.ML.Data;

namespace InnAi.Model.Models;

public class ModelInput
{
    [ColumnName(nameof(PrecipitationMap))]
    [VectorType(1024)]
    public float[] PrecipitationMap;
    
    [ColumnName(nameof(InnLevels))]
    [VectorType(9)]
    public float[] InnLevels { get; set; }
    
    [VectorType(48)]
    public float[] Predictions { get; set; }
}