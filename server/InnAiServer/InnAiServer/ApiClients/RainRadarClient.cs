using InnAiServer.Options;
using Microsoft.Extensions.Options;
using System.Text.Json;
using InnAiServer.Models.RainViewer;

namespace InnAiServer.ApiClients;

public class RainRadarClient : IRainRadarClient
{
    private readonly ILogger<RainRadarClient> _logger;
    private readonly HttpClient _client;

    public RainRadarClient(ILogger<RainRadarClient> logger, IOptions<RainRadarOptions> rainRadarOptions)
    {
        _logger = logger;
        _client = new HttpClient
        {
            BaseAddress = new Uri(rainRadarOptions.Value.ApiBaseUrl?.TrimEnd('/') ?? throw new Exception())
        };
    }
    
    public async Task<IEnumerable<(byte[] Data, DateTime Timestamp)>> GetLatestRadarImageAsync(DateTime from)
    {
        const int size = 512;
        const int zoom = 8;
        const int x = 136;
        const int y = 89;
        const int color = 1;
        const string options = "1_1";
        
        var dateTimes =  (await GetLatestTimeAsync())
            .Where(x => UnixTimeStampToDateTime(x) > from);

        var result = new List<(byte[] Data, DateTime Timestamp)>();

        foreach (var time in dateTimes)
        {
            var response = await _client.GetAsync($"/v2/radar/{time}/{size}/{zoom}/{x}/{y}/{color}/{options}.png");
            response.EnsureSuccessStatusCode();

            var data = await response.Content.ReadAsByteArrayAsync();
            
            result.Add((data, UnixTimeStampToDateTime(time)));
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