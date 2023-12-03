namespace InnAiServer.Dtos;

public record InnAiDataDto(int Count, TrainingDataDto[] Items);
public record TrainingDataDto(DateTime Timestamp, InnLevelDto[] InnLevels, string ImageId);

public record InnLevelDto(int? Level, string Station);
