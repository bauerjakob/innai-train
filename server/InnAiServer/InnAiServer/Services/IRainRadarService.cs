using InnAiServer.Data.Collections;
using MongoDB.Bson;

namespace InnAiServer.Services;

public interface IRainRadarService
{
    public Task<ObjectId[]> GetLastIdAsync(int count);
    public Task<byte[]> GetRadarImageAsync(string radarId);
    public Task<double[,]> GetRadarImageValuesAsync(string radarId);
    // public Task DownloadAndStoreLatestRadarImagesAsync();

    public Task LoadMonthAsync(int year, int month);
    public Task<RainRadar> GetAsync(ObjectId rainRadarId);
}