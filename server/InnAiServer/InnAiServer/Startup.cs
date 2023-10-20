using Amazon.Runtime;
using InnAiServer.ApiClients;
using InnAiServer.Data.Repositories;
using InnAiServer.HostedServices;
using InnAiServer.Options;
using InnAiServer.Services;
using Microsoft.AspNetCore.Mvc.Versioning;
using Microsoft.OpenApi.Models;
using MongoDB.Driver;

namespace InnAiServer;

public class Startup
{
    private readonly IConfiguration _configuration;

    public Startup(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    private string? MongoConnectionString => _configuration.GetConnectionString("MongoDb"); 
    
    // This method gets called by the runtime. Use this method to add services to the container.
    public void ConfigureServices(IServiceCollection services)
    {
        services.AddControllers();

        services.AddApiVersioning(x =>
        {
            x.DefaultApiVersion = new Microsoft.AspNetCore.Mvc.ApiVersion(1,0);
            x.AssumeDefaultVersionWhenUnspecified = true;
            x.ReportApiVersions = true;
            x.ApiVersionReader = ApiVersionReader.Combine(new UrlSegmentApiVersionReader(),
                new HeaderApiVersionReader("x-api-version"),
                new MediaTypeApiVersionReader("x-api-version"));
        });

        ConfigureOptions(services);
        ConfigureCustomServices(services);
        ConfigureSwagger(services);
        ConfigureDatabase(services);
        ConfigureHostedServices(services);
    }

    private void ConfigureSwagger(IServiceCollection services)
    {
        services.AddSwaggerGen(options =>
        {
            options.SwaggerDoc("v1", new OpenApiInfo {
                Title = "InnAi API",
                Description = "InnAi API",
                Version = "v1",
                Contact = new OpenApiContact
                {
                    Name = "Jakob Bauer",
                    Url = new Uri("https://www.bauer-jakob.de"),
                    Email = "info@bauer-jakob.de"
                }
            });
        });
    }

    private void ConfigureOptions(IServiceCollection services)
    {
        services.Configure<RainRadarOptions>(_configuration.GetSection(nameof(RainRadarOptions)));
        services.Configure<InnLevelOptions>(_configuration.GetSection(nameof(InnLevelOptions)));
    }

    private void ConfigureDatabase(IServiceCollection services)
    {
        services.AddScoped<IMongoClient, MongoClient>(x => new MongoClient(MongoConnectionString));

        services.AddScoped<IMongoDatabase>(x =>
        {
            var mongoClient = x.GetRequiredService<IMongoClient>();
            return mongoClient.GetDatabase("InnAi");
        });

        services.AddScoped<IRainRadarRepository, RainRadarRepository>();
        services.AddScoped<IInnLevelRepository, InnLevelRepository>();
    }

    private void ConfigureCustomServices(IServiceCollection services)
    {
        services.AddScoped<IRainRadarService, RainRadarService>();
        services.AddScoped<IRainRadarClient, RainRadarClient>();
        services.AddScoped<IInnLevelService, InnLevelService>();
        services.AddScoped<IInnLevelClient, InnLevelClient>();
    }

    private void ConfigureHostedServices(IServiceCollection services)
    {
        services.AddHostedService<RainRadarHostedService>();
        services.AddHostedService<InnLevelHostedService>();
    }
    
    // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        // if (env.IsDevelopment())
        // {
        app.UseSwagger();
        app.UseSwaggerUI();
        // }

        app.UseHttpsRedirection();
        
        app.UseRouting();

        app.UseAuthentication();
        app.UseAuthorization();
        
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
        
        app.UseSwagger();

        app.UseSwaggerUI(options =>
        {
            options.SwaggerEndpoint("/swagger/v1/swagger.json", "InnAi API");
            // options.RoutePrefix = string.Empty;
        });
    }
}