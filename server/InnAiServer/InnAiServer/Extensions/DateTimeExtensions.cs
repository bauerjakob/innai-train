namespace InnAiServer.Extensions;

public static class DateTimeExtensions
{
    public static DateTime ToGermanTime(this DateTimeOffset dateTime)
    {
        var timeZone = TimeZoneInfo.FindSystemTimeZoneById("W. Europe Standard Time");
        var time = TimeZoneInfo.ConvertTime(dateTime, timeZone);
        return time.DateTime;
    }
}