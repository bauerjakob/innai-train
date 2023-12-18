namespace InnAi.Model.DataLoader;

public interface IDataLoader<T, K>
{
   public Task<T> LoadAsync(K data, CancellationToken cancellationToken);
}