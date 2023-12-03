using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using InnAiServer.Data.Collections;
using InnAiServer.Dtos;
using InnAiServer.Dtos.InnLevel;
using InnAiServer.Extensions;
using InnAiServer.Options;
using Microsoft.Extensions.Options;
using ZstdSharp.Unsafe;

namespace InnAiServer.ApiClients;

public class InnLevelClient : IInnLevelClient
{
    private readonly HttpClient _client;
    
    public InnLevelClient(IOptions<InnLevelOptions> innLevelOptions)
    {
        // https://api.pegelalarm.at/api/station/1.0/height/18001508-de/history?granularity=raw&loadEndDate=18.10.2023T11:00:00%2B0200&loadStartDate=17.10.2022T11:00:00%2B0200
        
        _client = new HttpClient
        {
            BaseAddress = new Uri(innLevelOptions.Value.ApiBaseUrl?.TrimEnd('/') ?? throw new Exception())
        };
    }

    // public Task<IEnumerable<InnLevel>?> GetLatestInnLevelsAsync(InnStation station, DateTimeOffset from)
    // {
    //     DateTimeOffset to = DateTime.UtcNow;
    //
    //     return GetLatestInnLevelsAsync(station, from, to);
    // }
    
    public async Task<IEnumerable<InnLevel>> GetInnLevelsAsync(InnStation station, DateTimeOffset from)
    {
        if (from > DateTimeOffset.Now)
        {
            throw new Exception();
        }
        
        // var endDate = ToUrlFriendlyTimeString(to.UtcDateTime.ToGermanTime());
        var startDate = ToUrlFriendlyTimeString(from.UtcDateTime.ToGermanTime());
        var requestUrl = $"/api/station/1.0/height/{station.Id}/history?granularity=raw&loadStartDate={startDate}"; // &loadEndDate={endDate}  
        
        var response = await _client.GetAsync(requestUrl);
        
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
    
        var result = json.Deserialize<PegelAlarmDto>(serializerOptions);
        return result?.Payload.History.Select(
            x => new InnLevel(UrlTimeToUtcDateTime(x.SourceDate), Convert.ToInt32(x.Value), station)) ?? new List<InnLevel>();
    }
    
    // public async Task<IEnumerable<InnLevel>> GetInnLevelsFromMonthAsync(InnStation station, int year, int month)
    // {
    //     var from = new DateTime(year, month, 1);
    //     var to = (new DateTime(year, month, DateTime.DaysInMonth(year, month))).AddDays(1);
    //     
    //     var data = await GetInnLevelsAsync(station, from, to);
    //
    //     return data ?? new List<InnLevel>();
    // }

    private static DateTime UrlTimeToUtcDateTime(string dateTime)
    {
        // var utcOffset = DateTimeExtensions.GetUtcOffset();
        var utcOffset = Regex.Match(dateTime, "(?<=\\+0)\\d").Value;
        var dateTimeOffset = DateTimeOffset.ParseExact(dateTime, $"dd.MM.yyyyTHH:mm:ss+0{utcOffset}00", new CultureInfo("de-DE"));
        return dateTimeOffset.UtcDateTime;
    }
    private static string ToUrlFriendlyTimeString(DateTime dateTime)
    {
        var utcOffset = dateTime.GetUtcOffset();
        return dateTime.ToString("dd.MM.yyyyTHH:mm:ss")+ $"%2B0{utcOffset}00";
    }
}