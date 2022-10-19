using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System.Collections.Generic;

namespace Matlabs.OwlRacer.MLNetClient
{
    public class OnnxInput
    {
        [ColumnName("Velocity")]
        public float Velocity { get; set; }

        [ColumnName("Distance_Front")]
        public long Distance_Front { get; set; }

        [ColumnName("Distance_FrontLeft")]
        public long Distance_FrontLeft { get; set; }

        [ColumnName("Distance_FrontRight")]
        public long Distance_FrontRight { get; set; }

        [ColumnName("Distance_Left")]
        public long Distance_Left { get; set; }

        [ColumnName("Distance_Right")]
        public long Distance_Right { get; set; }
    }
    public class OnnxOutput
    {
        [ColumnName("output_label")]
        public Int64[] Output_label { get; set; }

        [VectorType(1, 5), OnnxMapType(typeof(Int64), typeof(Single)), NoColumn]
        [ColumnName("output_probability")]
        public IEnumerable<IDictionary<Int64, float>>? Output_probability { get; set; }
    }
}
