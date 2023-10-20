namespace InnAiServer.Models.RainViewer;

public record RainViewerOverview(string version, int Generated, string Host, RainViewerOverviewRadar Radar);

public record RainViewerOverviewRadar(RainViewerOverviewRadarItem[] Past);

public record RainViewerOverviewRadarItem(int Time, string Path);