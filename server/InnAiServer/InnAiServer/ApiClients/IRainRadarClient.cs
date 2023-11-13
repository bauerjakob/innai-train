namespace InnAiServer.ApiClients;

public interface IRainRadarClient
{
    public Task<IEnumerable<(byte[] DataRainSnow, byte[] DataRain, DateTime Timestamp)>> GetLatestRadarImageAsync(DateTime from);
}