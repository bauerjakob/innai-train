namespace InnAiServer.Services;

public interface IInnLevelService
{
    public Task DownloadAndStoreLatestRadarImageAsync();
}