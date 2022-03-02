using System;
using System.IO;
using System.Reflection.Metadata;
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

        static string ONNX_MODEL_PATH_RandomForest = "owlracer-ai/src/PythonSamples/trainedModels/RandomForestClassifier_owlracer.onnx";
        static string ONNX_MODEL_PATH_DecisionTree = "owlracer-ai/src/PythonSamples/trainedModels/DecisionTreeClassifier_owlracer.onnx";
        static string ONNX_MODEL_PATH_KNeighbors = "devowlracer-ai/src/PythonSamples/trainedModels/KNeighborsClassifier_owlracer.onnx";
        static string ONNX_MODEL_PATH_LogisticRegression = "owlracer-ai/src/PythonSamples/trainedModels/LogisticRegression_owlracer.onnx";
        static string ONNX_MODEL_PATH_pipeline = "owlracer-ai/src/PythonSamples/trainedModels/pipeline_owlracer.onnx";
        
        static ITransformer GetPredictionPipeline(MLContext mlContext)
        {
            var inputColumns = new string[]
            {
                "Velocity", "Distance_Front", "Distance_FrontLeft", "Distance_FrontRight", "Distance_Left", "Distance_Right"
            };

            var outputColumns = new string[] { "output_label", "output_probability" };

            var onnxPredictionPipeline =
                mlContext
                    .Transforms
                    .ApplyOnnxModel(
                        outputColumnNames: outputColumns,
                        inputColumnNames: inputColumns,
                        _config.Model); //RandomForest, DecisionTree, KNeighbors, LogisticRegression, pipeline

            var emptyDv = mlContext.Data.LoadFromEnumerable(new OnnxInput[] { });

            return onnxPredictionPipeline.Fit(emptyDv);
        }
        public static async Task Main(string[] args)
        {
            ParseConfig(args);
            
            // Prepare some control variables and objects.
            GrpcCoreService.GrpcCoreServiceClient client = null;
            SessionData sessionData = null;
            RaceCarData carData = null;

            try
            {
                // Create channel and gRPC client.
                Console.WriteLine("Creating channel and gRPC client.");
                var grpcChannel = new Channel($"localhost:6003", ChannelCredentials.Insecure);
                client = new GrpcCoreService.GrpcCoreServiceClient(grpcChannel);

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

                ////Load Model 
                MLContext mlContext = new MLContext();

                var onnxPredictionPipeline = GetPredictionPipeline(mlContext);
                var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(onnxPredictionPipeline);

                while (true)
                {
                    var EngineInput = new OnnxInput { 
                        Velocity=carData.Velocity,
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

                    //if (carData.IsCrashed == true)
                    //{
                    //    client.Reset(carData.Id);
                    //}

                }
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
            else {
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

            _config = config;
        }
    }
}
