using InnAiServer.Options;
using Microsoft.Extensions.Options;
using System.Text.Json;
using InnAiServer.Models.RainViewer;

namespace InnAiServer.ApiClients;

public class RainRadarClient : IRainRadarClient
{
    private readonly ILogger<RainRadarClient> _logger;
    private readonly RainRadarOptions _rainRadarOptions;
    private readonly HttpClient _client;

    public RainRadarClient(ILogger<RainRadarClient> logger, IOptions<RainRadarOptions> rainRadarOptions)
    {
        _logger = logger;
        _rainRadarOptions = rainRadarOptions.Value;
        _client = new HttpClient
        {
            BaseAddress = new Uri(rainRadarOptions.Value.ApiBaseUrl?.TrimEnd('/') ?? throw new Exception())
        };
    }
    
    public async Task<IEnumerable<(byte[] DataRainSnow, byte[] DataRain, DateTime Timestamp)>> GetLatestRadarImageAsync(DateTime from)
    {
        const int size = 256;
        const int color = 0;
        const string optionsWithSnow = "0_1";
        const string optionsWithoutSnow = "0_0";
        
        var dateTimes =  (await GetLatestTimeAsync())
            .Where(x => UnixTimeStampToDateTime(x) > from);

        var result = new List<(byte[] DataRainSnow, byte[] DataRain, DateTime Timestamp)>();

        foreach (var time in dateTimes)
        {
            var responseWithSnow = await _client.GetAsync($"/v2/radar/{time}/{size}/{_rainRadarOptions.Zoom}/{_rainRadarOptions.X}/{_rainRadarOptions.Y}/{color}/{optionsWithSnow}.png");
            responseWithSnow.EnsureSuccessStatusCode();
            
            var responseWithoutSnow = await _client.GetAsync($"/v2/radar/{time}/{size}/{_rainRadarOptions.Zoom}/{_rainRadarOptions.X}/{_rainRadarOptions.Y}/{color}/{optionsWithSnow}.png");
            responseWithoutSnow.EnsureSuccessStatusCode();

            var dataWithSnow = await responseWithSnow.Content.ReadAsByteArrayAsync();
            var dataWithoutSnow = await responseWithoutSnow.Content.ReadAsByteArrayAsync();
            
            result.Add((dataWithSnow, dataWithoutSnow, UnixTimeStampToDateTime(time)));
        }

        return result;
    }


    private async Task<IEnumerable<int>> GetLatestTimeAsync()
    {
        var response = await _client.GetAsync("/public/weather-maps.json");
        if (!response.IsSuccessStatusCode)
        {
            throw new Exception();
        }

        var content = await response.Content.ReadAsStreamAsync();
        var json = await JsonDocument.ParseAsync(content);
        
        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        var rainViewerOverview = json.Deserialize<RainViewerOverview>(serializerOptions);
        
        var latestTime = rainViewerOverview?
            .Radar
            .Past
            .OrderByDescending(x => x.Time)
            .Select(x => x.Time);
        
        return latestTime.ToList();
    }
    
    private static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
    {
        var dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
        dateTime = dateTime.AddSeconds(unixTimeStamp).ToUniversalTime();
        return dateTime;
    }
}