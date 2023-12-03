// using InnAiServer.Options;
// using Microsoft.Extensions.Options;
// using System.Text.Json;
// using InnAiServer.Models;
// using InnAiServer.Models.RainViewer;
//
// namespace InnAiServer.ApiClients;
//
// public class RainViewerClient : IWeatherApiClient
// {
//     private readonly ILogger<RainViewerClient> _logger;
//     private readonly PrecipitationClientOptions _precipitationClientOptions;
//     private readonly HttpClient _client;
//
//     public RainViewerClient(ILogger<RainViewerClient> logger, IOptions<PrecipitationClientOptions> rainRadarOptions)
//     {
//         _logger = logger;
//         _precipitationClientOptions = rainRadarOptions.Value;
//         _client = new HttpClient
//         {
//             BaseAddress = new Uri(rainRadarOptions.Value.ApiBaseUrl?.TrimEnd('/') ?? throw new Exception())
//         };
//     }
//
//     public Task<PrecipitationData> GetLatestRadarImageAsync()
//     {
//         throw new NotImplementedException();
//     }
//
//     public async Task<IEnumerable<PrecipitationData>> GetLatestRadarImageAsync(DateTime from)
//     {
//         const int size = 256;
//         const int color = 0;
//         const string optionsWithSnow = "0_1";
//         const string optionsWithoutSnow = "0_0";
//         
//         var dateTimes =  (await GetLatestTimeAsync())
//             .Where(x => UnixTimeStampToDateTime(x) > from);
//
//         var result = new List<PrecipitationData>();
//
//         foreach (var time in dateTimes)
//         {
//             var responseWithoutSnow = await _client.GetAsync($"/v2/radar/{time}/{size}/{_precipitationClientOptions.Zoom}/{_precipitationClientOptions.X}/{_precipitationClientOptions.Y}/{color}/{optionsWithSnow}.png");
//             responseWithoutSnow.EnsureSuccessStatusCode();
//
//             var dataWithoutSnow = await responseWithoutSnow.Content.ReadAsByteArrayAsync();
//             
//             result.Add(new PrecipitationData(UnixTimeStampToDateTime(time), dataWithoutSnow));
//         }
//
//         return result;
//     }
//
//
//     private async Task<IEnumerable<int>> GetLatestTimeAsync()
//     {
//         var response = await _client.GetAsync("/public/weather-maps.json");
//         if (!response.IsSuccessStatusCode)
//         {
//             throw new Exception();
//         }
//
//         var content = await response.Content.ReadAsStreamAsync();
//         var json = await JsonDocument.ParseAsync(content);
//         
//         var serializerOptions = new JsonSerializerOptions
//         {
//             PropertyNameCaseInsensitive = true
//         };
//
//         var rainViewerOverview = json.Deserialize<RainViewerOverview>(serializerOptions);
//         
//         var latestTime = rainViewerOverview?
//             .Radar
//             .Past
//             .OrderByDescending(x => x.Time)
//             .Select(x => x.Time);
//         
//         return latestTime.ToList();
//     }
//     
//     private static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
//     {
//         var dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
//         dateTime = dateTime.AddSeconds(unixTimeStamp).ToUniversalTime();
//         return dateTime;
//     }
// }