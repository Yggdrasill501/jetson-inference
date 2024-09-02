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

from jetson_inference import backgroundNet
from jetson_utils import videoSource, videoOutput, loadImage, Log
from jetson_utils import cudaAllocMapped, cudaMemcpy, cudaResize, cudaOverlay

# Function to parse command line arguments
fn parse_args() -> (str, str, str, str, str):
    parser = argparse.ArgumentParser(description="Perform background subtraction/removal and replacement.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=backgroundNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="u2net", help="pre-trained model to load (see below for options)")
    parser.add_argument("--replace", type=str, default="", help="image filename to use for background replacement")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input_URI, args.output_URI, args.network, args.replace, args.filter_mode

# Image replacement routine
fn replace_background(img_input, img_replacement, img_replacement_scaled, img_output, filter_mode: str) -> Any:
    if img_replacement_scaled is None or img_input.shape != img_replacement_scaled.shape:
        img_replacement_scaled = cudaAllocMapped(like=img_input)
        img_output = cudaAllocMapped(like=img_input)
        cudaResize(img_replacement, img_replacement_scaled, filter=filter_mode)

    cudaMemcpy(img_output, img_replacement_scaled)
    cudaOverlay(img_input, img_output, 0, 0)

    return img_output

# Main function
fn main():
    input_URI, output_URI, network, replace_image, filter_mode = parse_args()

    # Load the background removal network
    net = backgroundNet(network, sys.argv)

    # Create video sources & outputs
    input_source = videoSource(input_URI, argv=sys.argv)
    output_sink = videoOutput(output_URI, argv=sys.argv)

    # Image replacement routines
    img_replacement = loadImage(replace_image, format='rgba8') if replace_image else None
    img_replacement_scaled = None
    img_output = None

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image (with alpha channel)
        img_input = input_source.Capture(format='rgba8')

        if img_input is None:
            continue

        # Perform background removal
        net.Process(img_input, filter=filter_mode)

        # Perform background replacement
        if replace_image:
            img_output = replace_background(img_input, img_replacement, img_replacement_scaled, img_output, filter_mode)
        else:
            img_output = img_input

        # Render the image
        output_sink.Render(img_output)

        # Update the title bar
        output_sink.SetStatus("backgroundNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        # Print out performance info
        net.PrintProfilerTimes()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

# Call the main function
if __name__ == "__main__":
    main()
