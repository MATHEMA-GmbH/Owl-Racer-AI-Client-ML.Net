using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Matlabs.OwlRacer.MLNetClient
{
    internal class MLNetException:Exception
    {
        [StackTraceHidden]
        public MLNetException()
        {
        }

        [StackTraceHidden]
        public MLNetException(string message)
            : base(message)
        {
        }

        [StackTraceHidden]
        public MLNetException(string message, Exception inner)
            : base(message, inner)
        {
        }

    }
}
