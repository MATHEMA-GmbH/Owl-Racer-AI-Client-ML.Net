using Microsoft.ML.Data;

namespace Matlabs.OwlRacer.MLNetClient
{
    public class NewOnnxInput
    {
        [ColumnName("input")]
        [VectorType(6, 1)]
        public float[] Input { get; set; }
    }
}
