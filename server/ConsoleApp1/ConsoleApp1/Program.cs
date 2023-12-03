using System.Globalization;

for (int i = 0; i < int.Parse("FF", NumberStyles.HexNumber); i++)
{
    Console.WriteLine(i * 6 / int.Parse("FC", NumberStyles.HexNumber));
}