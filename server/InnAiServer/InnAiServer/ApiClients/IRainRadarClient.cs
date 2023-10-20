namespace InnAiServer.ApiClients;

public interface IRainRadarClient
{
    public Task<IEnumerable<(byte[] Data, DateTime Timestamp)>> GetLatestRadarImageAsync(DateTime from);
}