namespace InnAi.Core;

public record TrainingData(int Count, TrainingDataItem[] Items);
public record TrainingDataItem(DateTime Timestamp, InnLevel[] InnLevels, string PrecipitationMapId, double[][] PrecipitationMapLarge, double[][] PrecipitationMapMedium, double[][] PrecipitationMapSmall, NextInnLevel[] NextInnLevels);

public record InnLevel(int? Level, string Station);
public record NextInnLevel(int? Level, int HoursOffset);
