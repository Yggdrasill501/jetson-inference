"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import sys
import argparse

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

# Function to parse command line arguments
fn parse_args() -> (str, str, str, str, float):
    parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
    parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input, args.output, args.network, args.overlay, args.threshold

# Main function
fn main():
    input_URI, output_URI, network, overlay, threshold = parse_args()

    # Load the pose estimation model
    net = poseNet(network, sys.argv, threshold)

    # Create video sources & outputs
    input_source = videoSource(input_URI, argv=sys.argv)
    output_sink = videoOutput(output_URI, argv=sys.argv)

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image
        img = input_source.Capture()

        if img is None:
            continue

        # Perform pose estimation (with overlay)
        poses = net.Process(img, overlay=overlay)

        # Print the pose results
        print("detected {:d} objects in image".format(len(poses)))

        for pose in poses:
            print(pose)
            print(pose.Keypoints)
            print('Links', pose.Links)

        # Render the image
        output_sink.Render(img)

        # Update the title bar
        output_sink.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))

        # Print out performance info
        net.PrintProfilerTimes()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

if __name__ == "__main__":
    main()
