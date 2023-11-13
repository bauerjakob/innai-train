using System.Globalization;
using System.Text;
using CsvHelper;
using InnAiServer.ApiClients;
using InnAiServer.Data.Collections;
using InnAiServer.Data.Repositories;
using InnAiServer.Models;
using ImageMagick;

namespace InnAiServer.Services;

public class RainRadarService : IRainRadarService
{
    private readonly ILogger<RainRadarService> _logger;
    private readonly IRainRadarClient _rainRadarClient;
    private readonly IRainRadarRepository _dbRepository;

    private IEnumerable<RainViewerColorSchemeItem>? _colorSchemeItems = null;

    public RainRadarService(ILogger<RainRadarService> logger, IRainRadarClient rainRadarClient, IRainRadarRepository dbRepository)
    {
        _logger = logger;
        _rainRadarClient = rainRadarClient;
        _dbRepository = dbRepository;
    }
    
    public Task<RainRadar[]> GetLastAsync(int count)
    {
        return _dbRepository.GetLastAsync(count);
    }

    public async Task<byte[]> GetRadarImageAsync(string radarId)
    {
        var radarItem = await _dbRepository.GetAsync(radarId);
        return radarItem.ImageRain;
    }

    public async Task<int[,]> GetRadarImageDbzAsync(string radarId)
    {
        var radarItem = await _dbRepository.GetAsync(radarId);
        return radarItem.DbzRain;
    }

    public async Task DownloadAndStoreLatestRadarImagesAsync()
    {
        var items = await DownloadLatestRadarImagesFromApiAsync();
        
        foreach (var rainData in items)
        {
            await _dbRepository.CreateAsync(rainData);
        }
        
        _logger.LogInformation("[{ServiceName}] Successfully downloaded and stored {Count} items", nameof(RainRadarService), items?.Count() ?? 0);
    }
    
    private async Task<IEnumerable<RainRadar>> DownloadLatestRadarImagesFromApiAsync()
    {
        var latestItem = (await _dbRepository.GetLastAsync(1)).SingleOrDefault();

        var latestTime = latestItem?.Timestamp ?? DateTime.MinValue;
        
        var data = await _rainRadarClient.GetLatestRadarImageAsync(latestTime);

        return data
            .Select(x =>
                new RainRadar(x.Timestamp,
                    x.DataRain,
                    x.DataRainSnow,
                    ImageToDbzValues(x.DataRain),
                    ImageToDbzValues(x.DataRainSnow)));
    }
    
    private async Task<RainRadar?> GetLastAsync()
    {
        var latestItem = (await GetLastAsync(1)).SingleOrDefault();
        return latestItem;
    }

    private int[,] ImageToDbzValues(byte[] imageData)
    {
        var image = new MagickImage(imageData, MagickFormat.Png);
        var pixels = image.GetPixels();
        var width = image.Width;
        var height = image.Height;
        
        var dbzValues = new int[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = pixels.GetPixel(x, y);
                var color = pixel.ToColor();
                var hexColor = color?.ToHexString();
                var dbz = ReadDbzValue(hexColor);
                dbzValues[y, x] = dbz;
            }
        }

        return dbzValues;
    }

    private int ReadDbzValue(string hexColor)
    {
        if (_colorSchemeItems == null)
        {
            using var reader = new StreamReader(Path.Combine("Resources", "RainViewerColorScheme.csv"), Encoding.UTF8);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            _colorSchemeItems = csv.GetRecords<RainViewerColorSchemeItem>().ToArray();
        }

        if (_colorSchemeItems == null)
        {
            throw new Exception();
        }


        var hexColorExtended = $"{hexColor}{(hexColor.Length == 7 ? "ff" : string.Empty)}";
        
        return _colorSchemeItems.Single(x => x.BlackWhite.Equals(hexColorExtended, StringComparison.OrdinalIgnoreCase)).Dbz;
    }
}