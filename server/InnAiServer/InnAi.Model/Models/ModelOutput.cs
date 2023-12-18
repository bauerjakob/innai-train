using Microsoft.ML.Data;

namespace InnAi.Model.Models;

public class ModelOutput
{
    [ColumnName("Predictions")]
    public float[] Predictions { get; set; }
    
    // public float[] Score { get; set; }
}