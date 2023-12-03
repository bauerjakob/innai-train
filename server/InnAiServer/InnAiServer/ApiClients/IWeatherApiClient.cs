using InnAiServer.Models;

namespace InnAiServer.ApiClients;

public interface IWeatherApiClient
{
    // public Task<PrecipitationData> GetLatestRadarImageAsync();
    // public Task<IEnumerable<PrecipitationData>> GetLatestRadarImageAsync(DateTime from);
    public Task<IEnumerable<PrecipitationData>> PrecipitationDataFromMonthAsync(int year, int month);

}