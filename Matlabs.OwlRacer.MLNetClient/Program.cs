﻿using System;
using System.IO;
using System.Threading.Tasks;
using Grpc.Core;
using Matlabs.OwlRacer.Protobuf;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;

namespace Matlabs.OwlRacer.MLNetClient
{
    public static class Program
    {
        private static Config _config;

        public static async Task Main(string[] args)
        {
            ParseConfig(args);

            // Prepare some control variables and objects.
            GrpcCoreService.GrpcCoreServiceClient client = null;
            RaceCarData carData = null;

            try
            {
                // Create channel and gRPC client.
                Console.WriteLine("Creating channel and gRPC client.");
                var grpcChannel = new Channel($"localhost:6003", ChannelCredentials.Insecure);
                client = new GrpcCoreService.GrpcCoreServiceClient(grpcChannel);

                SessionData sessionData;
                if (_config.SessionId == null)
                {
                    Console.WriteLine("Creating session...");
                    sessionData = await client.CreateSessionAsync(new CreateSessionData
                    {
                        GameTimeSetting = 40f,
                        TrackNumber = 2,
                        Name = "ML.NET"
                    });
                }
                else
                {
                    Console.WriteLine("Joining session...");
                    sessionData = await client.GetSessionAsync(new GuidData
                    {
                        GuidString = _config.SessionId.Value.ToString()
                    });
                }

                Console.WriteLine($"Using session with ID: {sessionData.Id}");

                // Create race car.
                Console.WriteLine("Creating race car...");
                carData = await client.CreateCarAsync(new CreateCarData
                {
                    Name = _config.CarName,
                    Color = _config.CarColor,
                    SessionId = sessionData.Id,
                    Acceleration = 0.05f,
                    MaxVelocity = 0.5f
                });
                Console.WriteLine($"Created car with ID: {carData.Id}");
                await RunModel(client, carData);
            }
            catch (Exception e)
            {
                await Console.Error.WriteLineAsync($"Oh noes! An unexpected error occurred: {e}");
            }
            finally
            {
                try
                {
                    if (carData != null)
                    {
                        Console.WriteLine("Destroying car...");
                        await client?.DestroyCarAsync(carData.Id);
                    }
                }
                catch (Exception e)
                {
                    await Console.Error.WriteLineAsync($"An error occured during final shutdown: {e}");
                }
            }

            await Task.CompletedTask;
        }

        private static void ParseConfig(string[] args)
        {
            var builder = new ConfigurationBuilder();
            builder.AddCommandLine(args);

            var cmdLineConfig = builder.Build();

            var sessionString = cmdLineConfig["session"];
            var trackString = cmdLineConfig["track"];
            var model = cmdLineConfig["model"];
            var carName = cmdLineConfig["carName"];
            var carColor = cmdLineConfig["carColor"];
            var versionString = cmdLineConfig["version"];

            var config = new Config();

            if (string.IsNullOrWhiteSpace(model))
            {
                Console.Error.WriteLine("Unable to start ML.NET without a valid model name!");
                Environment.Exit(1);
            }

            if (!File.Exists(model))
            {
                Console.Error.WriteLine($"Unable to run model '{model}', the file does not exist!");
                Environment.Exit(1);
            }

            config.Model = model;

            if (Guid.TryParse(sessionString, out var sessionGuid))
            {
                config.SessionId = sessionGuid;
            }

            if (int.TryParse(trackString, out var trackNumber))
            {
                config.TrackNumber = trackNumber;
            }

            if (string.IsNullOrWhiteSpace(carName))
            {
                config.CarName = "ML.NET";
            }
            else
            {
                config.CarName = carName;
            }

            if (string.IsNullOrWhiteSpace(carColor))
            {
                config.CarColor = "#bd11b4";//string.Empty;
            }
            else
            {
                config.CarColor = carColor;
            }

            if (int.TryParse(versionString, out var version))
            {
                config.Version = version;
            }

            _config = config;
        }

        static ITransformer GetPredictionPipeline(MLContext mlContext)
        {
            string[] inputColumns;

            if (_config.Version == (int)Version.VersionOne)
            {
                inputColumns = new string[]
               {
                    "input"
               };
            }
            else
            {
                inputColumns = new string[]
                {
                    "Velocity",
                    "Distance_Front",
                    "Distance_FrontLeft",
                    "Distance_FrontRight",
                    "Distance_Left",
                    "Distance_Right"
                };

            }

            var outputColumns = new string[] { "output_label", "output_probability" };

            var onnxPredictionPipeline =
                mlContext
                    .Transforms
                    .ApplyOnnxModel(
                        outputColumnNames: outputColumns,
                        inputColumnNames: inputColumns,
                        _config.Model); //RandomForest, DecisionTree, KNeighbors, LogisticRegression, pipeline

            if (_config.Version == (int)Version.VersionOne) return onnxPredictionPipeline.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<NewOnnxInput>()));

            return onnxPredictionPipeline.Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<OnnxInput>()));
        }

        private static async Task RunModel(GrpcCoreService.GrpcCoreServiceClient client, RaceCarData carData)
        {
            ////Load Model 
            MLContext mlContext = new();

            var onnxPredictionPipeline = GetPredictionPipeline(mlContext);

            if (_config.Version == (int)Version.VersionOne)
            {
                var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<NewOnnxInput, OnnxOutput>(onnxPredictionPipeline);

                while (true)
                {
                    var engineInput = new NewOnnxInput
                    {
                        Input = new float[] {
                            carData.Velocity,
                            carData.Distance.Front,
                            carData.Distance.FrontLeft,
                            carData.Distance.FrontRight,
                            carData.Distance.Left,
                            carData.Distance.Right
                        }
                    };

                    var prediction = onnxPredictionEngine.Predict(engineInput);

                    carData = await client.GetCarDataAsync(carData.Id);
                    carData = await client.StepAsync(new StepData
                    {
                        CarId = carData.Id,
                        Command = (StepData.Types.StepCommand)prediction.Output_label[0]
                    });
                }
            }
            else
            {
                var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(onnxPredictionPipeline);

                while (true)
                {
                    var EngineInput = new OnnxInput
                    {
                        Velocity = carData.Velocity,
                        Distance_Front = carData.Distance.Front,
                        Distance_FrontLeft = carData.Distance.FrontLeft,
                        Distance_FrontRight = carData.Distance.FrontRight,
                        Distance_Left = carData.Distance.Left,
                        Distance_Right = carData.Distance.Right
                    };
                    var prediction = onnxPredictionEngine.Predict(EngineInput);


                    carData = await client.GetCarDataAsync(carData.Id);
                    carData = await client.StepAsync(new StepData
                    {
                        CarId = carData.Id,
                        Command = (StepData.Types.StepCommand)prediction.Output_label[0]
                    });
                }
            }
        }
    }
}
