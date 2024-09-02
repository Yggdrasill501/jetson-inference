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

import argparse
import datetime
import math
import sys
import os

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, saveImage, Log
from jetson_utils import cudaAllocMapped, cudaCrop, cudaDeviceSynchronize

# Function to parse command line arguments
fn parse_args() -> (str, str, str, str, float, str, str):
    parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
    parser.add_argument("--snapshots", type=str, default="images/test/detections", help="output directory of detection snapshots")
    parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S-%f", help="timestamp format used in snapshot filenames")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    return args.input, args.output, args.network, args.overlay, args.threshold, args.snapshots, args.timestamp

# Main function
fn main():
    input_URI, output_URI, network, overlay, threshold, snapshots_dir, timestamp_format = parse_args()

    # Make sure the snapshots dir exists
    os.makedirs(snapshots_dir, exist_ok=True)

    # Create video output object
    output_sink = videoOutput(output_URI, argv=sys.argv)

    # Load the object detection network
    net = detectNet(network, sys.argv, threshold)

    # Create video source
    input_source = videoSource(input_URI, argv=sys.argv)

    # Process frames until EOS or the user exits
    while True:
        # Capture the next image
        img = input_source.Capture()

        if img is None:
            continue

        # Detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=overlay)

        # Print the detections
        print("detected {:d} objects in image".format(len(detections)))

        timestamp = datetime.datetime.now().strftime(timestamp_format)

        for idx, detection in enumerate(detections):
            print(detection)
            roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            snapshot = cudaAllocMapped(width=roi[2]-roi[0], height=roi[3]-roi[1], format=img.format)
            cudaCrop(img, snapshot, roi)
            cudaDeviceSynchronize()
            saveImage(os.path.join(snapshots_dir, f"{timestamp}-{idx}.jpg"), snapshot)
            del snapshot

        # Render the image
        output_sink.Render(img)

        # Update the title bar
        output_sink.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))

        # Print out performance info
        net.PrintProfilerTimes()

        # Exit on input/output EOS
        if not input_source.IsStreaming() or not output_sink.IsStreaming():
            break

# Call the main function
if __name__ == "__main__":
    main()
