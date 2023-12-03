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
    private readonly IWeatherApiClient _weatherApiClient;
    private readonly IRainRadarRepository _dbRepository;

    private IEnumerable<OpenWeatherMapColorSchemeItem>? _colorSchemeItems = null;

    public RainRadarService(ILogger<RainRadarService> logger, IWeatherApiClient weatherApiClient, IRainRadarRepository dbRepository)
    {
        _logger = logger;
        _weatherApiClient = weatherApiClient;
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

    public async Task<int[,]> GetRadarImageValuesAsync(string radarId)
    {
        var radarItem = await _dbRepository.GetAsync(radarId);
        return radarItem.ValuesRainReducedMin;
    }

    // public async Task DownloadAndStoreLatestRadarImagesAsync()
    // {
    //     var items = await DownloadLatestRadarImagesFromApiAsync();
    //     
    //     foreach (var rainData in items)
    //     {
    //         await _dbRepository.CreateAsync(rainData);
    //     }
    //     
    //     _logger.LogInformation("[{ServiceName}] Successfully downloaded and stored {Count} items", nameof(RainRadarService), items?.Count() ?? 0);
    // }

    public async Task LoadMonthAsync(int year, int month)
    {
        var data = await _weatherApiClient.PrecipitationDataFromMonthAsync(year, month);

        foreach (var item in data)
        {
            var values = ImageToValues(item.Data);
            var valuesReducesMin = ReduceImageValuesAsync(values, x => x.Min());
            var valuesReducesMax = ReduceImageValuesAsync(values, x => x.Max());
            var valuesReducesAvg = ReduceImageValuesAsync(values, x => (int)x.Average());

            var rainData = new RainRadar(item.Timestamp, item.Data, values, valuesReducesMin, valuesReducesMax, valuesReducesAvg);
            await _dbRepository.CreateAsync(rainData);
        }

        _logger.LogInformation("[{ServiceName}] year={year} month={month} Successfully downloaded and stored {Count} items", nameof(RainRadarService), year, month, data?.Count() ?? 0);
    }

    // private async Task<IEnumerable<RainRadar>> DownloadLatestRadarImagesFromApiAsync()
    // {
    //     var latestItem = (await _dbRepository.GetLastAsync(1)).SingleOrDefault();
    //
    //     var latestTime = latestItem?.Timestamp ?? DateTime.MinValue;
    //     
    //     var data = await _weatherApiClient.GetLatestRadarImageAsync(latestTime);
    //
    //     return data
    //         .Select(x =>
    //             new RainRadar(x.Timestamp,
    //                 x.Data,
    //                 ImageToDbzValues(x.Data),
    //                 ImageToDbzValues(x.Data)));
    // }
    
    private async Task<RainRadar?> GetLastAsync()
    {
        var latestItem = (await GetLastAsync(1)).SingleOrDefault();
        return latestItem;
    }

    private int[,] ImageToValues(byte[] imageData)
    {
        var image = new MagickImage(imageData, MagickFormat.Png);
        
        var pixels = image.GetPixels();
        var width = image.Width;
        var height = image.Height;
        
        var values = new int[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = pixels.GetPixel(x, y);
                var color = pixel.ToColor();
                var hexColor = color?.ToHexString();
                var value = ReadValueAsync(hexColor);
                values[y, x] = value;
            }
        }

        return values;
    }

    private int ReadValueAsync(string hexColor)
    {
        if (_colorSchemeItems == null)
        {
            // using var reader = new StreamReader(Path.Combine("Resources", "OpenWeatherMapColorScheme.csv"), Encoding.UTF8);
            // using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            // _colorSchemeItems = csv.GetRecords<OpenWeatherMapColorSchemeItem>().ToArray();
        }

        // if (_colorSchemeItems == null)
        // {
        //     throw new Exception();
        // }
        
        var hexColorExtended = $"{hexColor}{(hexColor.Length == 7 ? "ff" : string.Empty)}";

        var hexColorUnextended = hexColorExtended.Substring(1, 2);
        
        var hexValue = (int)Math.Round(int.Parse(hexColorUnextended, NumberStyles.HexNumber) * 6.0 / int.Parse("FC", NumberStyles.HexNumber), 0, MidpointRounding.ToEven);  
        
        return hexValue;
    }


    private int[,] ReduceImageValuesAsync(int[,] imageData, Func<int[], int> selector)
    {
        int pixels = 8;
        var size = imageData.GetLength(0) / pixels;

        var ret = new int[size, size];

        for (var i = 0; i < size; i++)
        {
            for (var j = 0; j < size; j++)
            {
                List<int> values = new();
                for (var x = 0; x < pixels; x++)
                {
                    for (var y = 0; y < pixels; y++)
                    { 
                        var value = imageData[i * pixels + x, j * pixels + y];
                        values.Add(value);
                    }
                }

                ret[i, j] = selector(values.ToArray());
            }
        }

        return ret;
    }
}