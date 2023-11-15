using InnAiServer.Data.Collections;

namespace InnAiServer.Options;

public class InnLevelOptions
{
    public string? ApiBaseUrl { get; set; }
    public InnStation[] Stations { get; set; }
}