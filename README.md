
## Image Processor Framework

### Description

This project contains a framework to read a set of infrared image frames from disk, processes
them through various image processing algorithms, and display them on the screen while the
algorithms are running. The project contains a few algorithms implemented in both CUDA and C++.
Currently only 16 bit PNG files are supported.

### Requirements

```
libpng
cuda
g++ with c++17 support
```

### Contents

```
cfg/              image_processor example configuration file
data/             image_processor example dataset
include/          external headers
src/              internal source code
    cpu/          cpu algorithm implementations
    cuda/         gpu algorithm implementations
    framework/    image_processor framework source code
```

### Building

```
make install
```

### Usage

```
cd bin
./image_processor <path/to/config/file> [Log level: (ERROR,WARN,INFO,DEBUG,TRACE)]
(default log lvel = INFO)
```

To run example program:

```
make
cd bin
./image_processor ../cfg/bats.txt INFO
```

#### Config Files

The configuration file is a simple text file with each line following the format:

```
PARAM_NAME=PARAM_VAL
# comment

```

Required parameters:
```
INPUT_DIR=path/to/dir/with/png/files    Directory should contain png files, they will be
                                        read in in string sorted order.
                                        Example [img1.png, img2.png, img3.png]
DISPLAY_TYPE=(All/FIRST_LAST/NONE)      ALL = display images after each algorithm
                                        FIRST_LAST = display original images and fully processed images
                                        NONE = No image displays
DELAY=0.0                               Add a delay in seconds after each image update, use 0.0 for no delay
```

See algorithm headers for additional necessary parameters
