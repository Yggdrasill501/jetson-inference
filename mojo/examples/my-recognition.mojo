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

from jetson_inference import imageNet
from jetson_utils import loadImage

import argparse

# Function to parse command line arguments
fn parse_args() -> (str, str):
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=str, help="filename of the image to process")
    parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, etc.")

    args = parser.parse_args()

    return args.filename, args.network

# Main function
fn main():
    filename, network = parse_args()

    # Load an image (into shared CPU/GPU memory)
    img = loadImage(filename)

    # Load the recognition network
    net = imageNet(network)

    # Classify the image
    class_idx, confidence = net.Classify(img)

    # Find the object description
    class_desc = net.GetClassDesc(class_idx)

    # Print out the result
    print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

if __name__ == "__main__":
    main()
