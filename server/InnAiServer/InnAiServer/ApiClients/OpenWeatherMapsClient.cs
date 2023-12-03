using InnAiServer.Extensions;
using InnAiServer.Models;
using InnAiServer.Options;
using Microsoft.Extensions.Options;
using MongoDB.Driver.Core.Operations;

namespace InnAiServer.ApiClients;

public class OpenWeatherMapsClient : IWeatherApiClient
{
    private readonly ILogger<OpenWeatherMapsClient> _logger;
    private readonly PrecipitationClientOptions _options;
    private readonly HttpClient _client;

    public OpenWeatherMapsClient(ILogger<OpenWeatherMapsClient> logger, IOptions<PrecipitationClientOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _client = new HttpClient
        {
            BaseAddress = new Uri(options.Value.ApiBaseUrl?.TrimEnd('/') ?? throw new Exception())
        };
    }
    
    public async Task<PrecipitationData> GetLatestRadarImageAsync()
    {
        var url = GetPrecipitationBaseUrl();
        
        var response = await _client.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var imageData = await response.Content.ReadAsByteArrayAsync();

        return new PrecipitationData(DateTime.UtcNow, imageData);
    }

    public Task<IEnumerable<PrecipitationData>> GetLatestRadarImageAsync(DateTime from)
    {
        throw new NotImplementedException();
    }

    public async Task<IEnumerable<PrecipitationData>> PrecipitationDataFromDayAsync(DateTime dateTime)
    {
        List<PrecipitationData> ret = new();
        
        var date = dateTime.Date;

        for (var i = 0; i < 24; i++)
        {
            if (i > 0)
            {
                date = date.AddHours(1);    
            }

            var unixTimeStamp = date.ToUnixTimeStamp();

            var url = GetPrecipitationBaseUrl() + $"&date={unixTimeStamp}";

            try
            {
                var result = await _client.GetAsync(url);
                result.EnsureSuccessStatusCode();
                var data = await result.Content.ReadAsByteArrayAsync();
            
                ret.Add(new PrecipitationData(date, data));
            }
            catch (Exception e)
            {
                _logger.LogError(e, string.Empty);
                continue;
            }
        }

        return ret;
    }

    public async Task<IEnumerable<PrecipitationData>> PrecipitationDataFromMonthAsync(int year, int month)
    {
        List<PrecipitationData> ret = new();

        var daysCount = 3; // DateTime.DaysInMonth(year, month);

        for (var i = 0; i < daysCount; i++)
        {
            var data = await PrecipitationDataFromDayAsync(new DateTime(year, month, i + 1));
            ret.AddRange(data);
        }

        return ret;
    }

    private string GetPrecipitationBaseUrl() =>
        $"/maps/2.0/weather/PAR0/{_options.Zoom}/{_options.X}/{_options.Y}?fill_bound=false&opacity=1&appid={_options.ApiKey}" +
        $"&palette=0:000000;0.1:2A2A2A;0.2:545454;0.5:7E7E7E;1:A8A8A8;10:D2D2D2;140:FCFCFC";
}