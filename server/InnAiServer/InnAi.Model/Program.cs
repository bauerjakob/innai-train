// See https://aka.ms/new-console-template for more information

using InnAi.Core;
using InnAi.Model;
using InnAi.Model.DataLoader;
using InnAi.Model.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

// string dataPath = Path.Combine(Environment.CurrentDirectory, "handwritten_digits_large.csv");
var dataPath = "/Users/jakobbauer/Downloads/d438edf0-d01a-4b44-80a2-6d5e75e12e5b.json";
  
var context = new MLContext();

var dataLoader = new FileDataLoader<TrainingData>();
var data = await dataLoader.LoadAsync(dataPath, CancellationToken.None);
var modelInput = data.Items.Select(x => x.ToModelInput());

// var dataSchema = SchemaDefinition.Create(typeof(TrainingDataItem));

var dataView = context.Data.LoadFromEnumerable(modelInput);

var partitions = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

var features = partitions.TrainSet.Schema.Select(x => x.Name).ToArray();
// pipeline

var pipeline = BuildTrainingPipeline(context);

// train the model
Console.WriteLine("Training model....");
var model = pipeline.Fit(partitions.TrainSet);

Console.WriteLine("Hello World");


IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
{
    // Data process configuration with pipeline data transformations 
    // var dataProcessPipeline = 
    //     mlContext.Transforms.Conversion.MapValueToKey("Levels", "InnLevels")
    //         // .Append(mlContext.Transforms.Conversion.MapValueToKey("Levels", "InnLevels"))
    //         .Append(mlContext.Transforms.Concatenate("Features", "PrecipitationMap"))
    //         .Append(mlContext.Transforms.Concatenate("Feature", "Features", "Levels"))
    //         .AppendCacheCheckpoint(mlContext);


    var pipeline1 = mlContext.Transforms
        .Concatenate("Features", "InnLevels", "PrecipitationMap")
        // .Append(mlContext.Transforms.Conversion.MapKeyToValue("Predictions"))
        .Append(mlContext.Regression.Trainers.Sdca("Predictions", "Features"));

    // // Set the training algorithm
    // var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "InnLevels", featureColumnName: "Feature")
    //     .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
    
    // var trainingPipeline = dataProcessPipeline.Append(trainer);

    return pipeline1;
}
