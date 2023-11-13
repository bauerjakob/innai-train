namespace InnAiServer.Extensions;

public static class DateTimeExtensions
{
    private static readonly TimeZoneInfo _germanTimeZone = TimeZoneInfo.FindSystemTimeZoneById("W. Europe Standard Time"); 
    public static DateTime ToGermanTime(this DateTime utcTime)
    {
        var time = TimeZoneInfo.ConvertTimeFromUtc(utcTime, _germanTimeZone);
        return time;
    }

    public static int GetUtcOffset()
    {
        return _germanTimeZone.GetUtcOffset(DateTime.UtcNow).Hours;
    }
}