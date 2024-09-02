#!/usr/bin/env mojo
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

from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log
from depthnet_utils import depthBuffers

# Function to parse command line arguments
fn parse_args() -> (str, str, str, str, float, str, str):
    parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
    parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
    parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
                                      choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted",
                                               "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input, args.output, args.network, args.visualize, args.depth_size, args.filter_mode, args.colormap

# Main function
fn main():
    input_URI, output_URI, network, visualize, depth_size, filter_mode, colormap = parse_args()

    # Load the segmentation network
    net = depthNet(network, sys.argv)

    # Create buffer manager
    buffers = depthBuffers(visualize, depth_size, filter_mode)

    # Create video sources & outputs
    input_source = videoSource(input_URI, argv=sys.argv)
    output_sink = videoOutput(output_URI, argv=sys.argv)

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image
        img_input = input_source.Capture()

        if img_input is None:
            continue

        # Allocate buffers for this size image
        buffers.Alloc(img_input.shape, img_input.format)

        # Process the mono depth and visualize
        net.Process(img_input, buffers.depth, colormap, filter_mode)

        # Composite the images
        if buffers.use_input:
            cudaOverlay(img_input, buffers.composite, 0, 0)

        if buffers.use_depth:
            cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)

        # Render the output image
        output_sink.Render(buffers.composite)

        # Update the title bar
        output_sink.SetStatus("{:s} | {:s} | Network {:.0f} FPS".format(network, net.GetNetworkName(), net.GetNetworkFPS()))

        # Print out performance info
        cudaDeviceSynchronize()
        net.PrintProfilerTimes()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

# Call the main function
if __name__ == "__main__":
    main()
