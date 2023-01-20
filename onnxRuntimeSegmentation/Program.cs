// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

const int IMAGE_WIDTH = 640;
const int IMAGE_HEIGHT = 640;

using Image<Rgb24> image = Image.Load<Rgb24>("asset\\img1.jpg", out IImageFormat format);

using Stream imageStream = new MemoryStream();
image.Mutate(x=>
{
    x.Resize(new ResizeOptions
    {
        Size = new Size(IMAGE_WIDTH, IMAGE_HEIGHT),
        Mode = ResizeMode.Crop
    }); ;
});

image.Save(imageStream, format);

var inputData = new DenseTensor<float>(new[] { 1, 3, IMAGE_WIDTH, IMAGE_HEIGHT });

image.ProcessPixelRows(pointer =>
{
    for (int y = 0; y < pointer.Height; y++)
    {
        Span<Rgb24> pixelSpan =  pointer.GetRowSpan(y);
        for (int x = 0; x < pointer.Width; x++)
        {
            inputData[0, 0, y, x] = pixelSpan[x].R / 255.0F;
            inputData[0, 1, y, x] = pixelSpan[x].R / 255.0F;
            inputData[0, 2, y, x] = pixelSpan[x].R / 255.0F;
        }
    }
});

var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("images", inputData)
};

using var session = new InferenceSession("asset\\best.onnx");

using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

var resultsArray = results.ToArray();

float[] outPutValue = resultsArray[0].AsEnumerable<float>().ToArray();
var slice531outPutValue = resultsArray[1].AsEnumerable<float>().ToArray();
var slice516outPutValue = resultsArray[4].AsEnumerable<float>().ToArray();

var multiDiementionOutput = new float[1, 25200, 38];
Buffer.BlockCopy(outPutValue, 0, multiDiementionOutput, 0, outPutValue.Length * sizeof(float));

var slice531MultiDimention = new float[1, 3, 80, 80, 38];
Buffer.BlockCopy(slice531outPutValue, 0, slice531MultiDimention, 0, slice531outPutValue.Length * sizeof(float));

var slice516MultiDimention = new float[1, 32, 160, 160];
Buffer.BlockCopy(slice516outPutValue, 0, slice516MultiDimention, 0, slice516outPutValue.Length * sizeof(float));


var test = multiDiementionOutput[0,0,0];

var outputpixel = new float[25200];

Rgba32 whiteColot = Color.White;

//for (int c = 0; c < 38; c++)
//{
//    using var outputImage = File.OpenWrite($"asset\\80out{c}.jpg");
//    using Image<Rgba32> slice531Image = new Image<Rgba32>(80, 80);
//    slice531Image.ProcessPixelRows(accessor =>
//    {
//        for (int y = 0; y < accessor.Height; y++)
//        {
//            Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

//            for (int x = 0; x < accessor.Width; x++)
//            {
//                ref Rgba32 pixel = ref pixelRow[x];

//                if (slice531MultiDimention[0, 0, y, x, c] < 0)
//                    pixel.R = 0;
//                else
//                    pixel.R = 255;
//                if (slice531MultiDimention[0, 1, y, x, c] < 0)
//                    pixel.G = 0;
//                else
//                    pixel.G = 255;
//                if (slice531MultiDimention[0, 2, y, x, c] < 0)
//                    pixel.B = 0;
//                else
//                    pixel.B = 255;

//            }
//        }
//    });
//    slice531Image.Save(outputImage, format);
//}

Console.WriteLine("80 done!");

for (int c = 0; c < 32; c++)
{
    using var outputImage = File.OpenWrite($"asset\\160out{c}.jpg");
    using Image<Rgba32> slice516Image = new Image<Rgba32>(160, 160);
    slice516Image.ProcessPixelRows(accessor =>
    {
        for (int y = 0; y < accessor.Height; y++)
        {
            Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

            for (int x = 0; x < accessor.Width; x++)
            {
                ref Rgba32 pixel = ref pixelRow[x];

                if (slice516MultiDimention[0, c, y, x] < 0)
                    pixel.R = 0;
                else
                    pixel.R = 255;
                if (slice516MultiDimention[0, c, y, x] < 0)
                    pixel.G = 0;
                else
                    pixel.G = 255;
                if (slice516MultiDimention[0, c, y, x] < 0)
                    pixel.B = 0;
                else
                    pixel.B = 255;
            }
        }
    });
    slice516Image.Save(outputImage, format);
}

Console.WriteLine("160 done!");
