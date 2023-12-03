namespace InnAiServer.Extensions;

public static class DateTimeExtensions
{
    private static readonly TimeZoneInfo _germanTimeZone = TimeZoneInfo.FindSystemTimeZoneById("W. Europe Standard Time"); 
    public static DateTime ToGermanTime(this DateTime utcTime)
    {
        var time = TimeZoneInfo.ConvertTimeFromUtc(utcTime, _germanTimeZone);
        return time;
    }

    public static int GetUtcOffset(this DateTime dateTime)
    {
        return _germanTimeZone.GetUtcOffset(dateTime).Hours;
    }

    public static long ToUnixTimeStamp(this DateTime dateTime)
        => ((DateTimeOffset)dateTime).ToUnixTimeSeconds();
}