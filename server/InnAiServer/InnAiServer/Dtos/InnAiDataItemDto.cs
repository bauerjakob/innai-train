namespace InnAiServer.Dtos;

public record InnAiDataDto(int Count, InnAiDataItemDto[] Items);
public record InnAiDataItemDto(DateTime Timestamp, InnLevelDto[] InnLevels, string ImageId);

public record InnLevelDto(int? Level, string Station);
