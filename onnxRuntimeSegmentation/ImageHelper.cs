using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnxRuntimeSegmentation
{
    internal class ImageHelper
    {
        public float[,,] LoadingImage(string path)
        {
            var image = Image.FromFile(path);

            var imageData = new float[3,640,640];
            var bitmap = new Bitmap(image);

            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    var pixel = bitmap.GetPixel(i, j);
                    imageData[0, i, j] = pixel.R;
                    imageData[1, i, j] = pixel.R;
                    imageData[2, i, j] = pixel.R;
                }
            }

            return imageData;
        }
    }
}
