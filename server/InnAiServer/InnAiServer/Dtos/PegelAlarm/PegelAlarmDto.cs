namespace InnAiServer.Dtos.InnLevel;

public record PegelAlarmDto(PegelAlarmDtoPayload Payload);

public record PegelAlarmDtoPayload(PegelAlarmDtoPayloadItem[] History);

public record PegelAlarmDtoPayloadItem(string SourceDate, double Value);