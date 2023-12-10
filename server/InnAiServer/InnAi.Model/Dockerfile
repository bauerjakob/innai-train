FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["InnAi.Model/InnAi.Model.csproj", "InnAi.Model/"]
RUN dotnet restore "InnAi.Model/InnAi.Model.csproj"
COPY . .
WORKDIR "/src/InnAi.Model"
RUN dotnet build "InnAi.Model.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "InnAi.Model.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "InnAi.Model.dll"]
