
#print("jetson.inference.__init__.mojo")

# jetson.inference links against jetson.utils, and it needs loaded
import jetson_utils

# load jetson.inference extension module
from jetson_inference_mojo import *

VERSION = '1.0.0'
