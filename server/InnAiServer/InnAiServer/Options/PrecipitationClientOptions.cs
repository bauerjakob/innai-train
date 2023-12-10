namespace InnAiServer.Options;

public class PrecipitationClientOptions
{
    public string? ApiBaseUrl { get; set; }
    public string? ApiKey { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    public int Zoom { get; set; }
    public string? ColorPalette { get; set; }
}