namespace InnAiServer.Models;

public record TrainingDataDto(int Count, TrainingDataItemDto[] Items);
public record TrainingDataItemDto(DateTime Timestamp, InnLevelDto[] InnLevels, string PrecipitationMapId, double[][] PrecipitationMapLarge, double[][] PrecipitationMapMedium, double[][] PrecipitationMapSmall, NextInnLevelDto[] NextInnLevels);

public record InnLevelDto(int? Level, string Station);
public record NextInnLevelDto(int? Level, int HoursOffset);
