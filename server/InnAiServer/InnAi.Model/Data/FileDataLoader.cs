
using System.Text.Json;
using System.Text.Json.Serialization;

namespace InnAi.Model.DataLoader;

public class FileDataLoader<T> : IDataLoader<T, string>
{
    public async Task<T> LoadAsync(string path, CancellationToken cancellationToken)
    {
        var fileStream = File.OpenRead(path);

        return await JsonSerializer.DeserializeAsync<T>(fileStream, cancellationToken: cancellationToken) ?? throw new Exception();
    }
}