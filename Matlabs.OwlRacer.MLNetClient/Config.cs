using System;

namespace Matlabs.OwlRacer.MLNetClient
{
    public class Config
    {
        public Guid? SessionId { get; set; }
        public int TrackNumber { get; set; }
        public string Model { get; set; } = string.Empty;
        public string CarName { get; set; } = string.Empty;
        public string CarColor { get; set; } = string.Empty;
        public int Version { get; set; } = default;
        public string Labelmap { get; set; } = string.Empty;
    }
}
