namespace InnAiServer.Dtos;

public record InnAiDataDto(int Count, InnAiDataItemDto[] Items);
public record InnAiDataItemDto(DateTime Timestamp, int Level, string ImageId);
