namespace InnAiServer.Dtos;

public record TrainingDataDto(int Count, TrainingDataItemDto[] Items);
public record TrainingDataItemDto(DateTime Timestamp, InnLevelDto[] InnLevels, string PrecipitationMapId, int[][] PrecipitationMapValues, NextInnLevelDto[] NextInnLevels);

public record InnLevelDto(int? Level, string Station);
public record NextInnLevelDto(int? Level, int HoursOffset);
