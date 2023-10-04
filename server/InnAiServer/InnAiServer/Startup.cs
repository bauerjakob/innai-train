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

    private string? MongoConnectionString => _configuration.GetConnectionString("DefaultConnection"); 
    
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


        
        
        
        
        ConfigureSwagger(services);
        ConfigureDatabase(services);
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

    private void ConfigureDatabase(IServiceCollection services)
    {
        services.AddSingleton<IMongoClient, MongoClient>(x => new MongoClient(MongoConnectionString));

        services.AddSingleton<IMongoDatabase>(x =>
        {
            var mongoClient = x.GetRequiredService<IMongoClient>();
            return mongoClient.GetDatabase("InnAi");
        });
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