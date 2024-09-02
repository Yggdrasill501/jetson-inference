#!/usr/bin/env mojo
"""
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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

from jetson_inference import actionNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# Function to parse command line arguments
fn parse_args() -> (str, str, str):
    parser = argparse.ArgumentParser(description="Classify the action/activity of an image sequence.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=actionNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="resnet-18", help="pre-trained model to load (see below for options)")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input, args.output, args.network

# Main function
fn main():
    input, output, network = parse_args()

    # Load the recognition network
    net = actionNet(network, sys.argv)

    # Create video sources & outputs
    input_source = videoSource(input, argv=sys.argv)
    output_sink = videoOutput(output, argv=sys.argv)
    font = cudaFont()

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image
        img = input_source.Capture()

        if img is None:
            continue

        # Classify the action sequence
        class_id, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_id)

        print(f"actionnet:  {confidence * 100:2.5f}% class #{class_id} ({class_desc})")

        # Overlay the result on the image
        font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)

        # Render the image
        output_sink.Render(img)

        # Update the title bar
        output_sink.SetStatus("actionNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        # Print out performance info
        net.PrintProfilerTimes()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

# Call the main function
if __name__ == "__main__":
    main()
