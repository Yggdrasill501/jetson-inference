"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log
from segnet_utils import segmentationBuffers

# Function to parse command line arguments
fn parse_args() -> (str, str, str, str, str, float, bool):
    parser = argparse.ArgumentParser(description="Segment a live camera stream using a semantic segmentation DNN.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
    parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
    parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
    parser.add_argument("--stats", type=bool, default=False, help="compute statistics about segmentation mask class output")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input, args.output, args.network, args.filter_mode, args.visualize, args.alpha, args.stats

# Main function
fn main():
    input_URI, output_URI, network, filter_mode, visualize, alpha, stats = parse_args()

    # Load the segmentation network
    net = segNet(network, sys.argv)

    # Set the alpha blending value
    net.SetOverlayAlpha(alpha)

    # Create video output
    output_sink = videoOutput(output_URI, argv=sys.argv)

    # Create buffer manager
    buffers = segmentationBuffers(net, visualize, alpha, filter_mode, stats)

    # Create video source
    input_source = videoSource(input_URI, argv=sys.argv)

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image
        img_input = input_source.Capture()

        if img_input is None:
            continue

        # Allocate buffers for this size image
        buffers.Alloc(img_input.shape, img_input.format)

        # Process the segmentation network
        net.Process(img_input, ignore_class=buffers.ignore_class)

        # Generate the overlay
        if buffers.overlay:
            net.Overlay(buffers.overlay, filter_mode=filter_mode)

        # Generate the mask
        if buffers.mask:
            net.Mask(buffers.mask, filter_mode=filter_mode)

        # Composite the images
        if buffers.composite:
            cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
            cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

        # Render the output image
        output_sink.Render(buffers.output)

        # Update the title bar
        output_sink.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))

        # Print out performance info
        cudaDeviceSynchronize()
        net.PrintProfilerTimes()

        # Compute segmentation class stats
        if stats:
            buffers.ComputeStats()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

if __name__ == "__main__":
    main()
