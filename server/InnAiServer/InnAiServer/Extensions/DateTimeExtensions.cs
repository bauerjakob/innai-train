namespace InnAiServer.Extensions;

public static class DateTimeExtensions
{
    public static DateTime ToGermanTime(this DateTime utcTime)
    {
        var timeZone = TimeZoneInfo.FindSystemTimeZoneById("W. Europe Standard Time");
        var time = TimeZoneInfo.ConvertTimeFromUtc(utcTime, timeZone);
        return time;
    }
}