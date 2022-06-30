# Gelsight ROS

This package provides an interface for using GelSight sensors in ROS.

As of right now, only R1.5 is supported.

## Dependencies

- \>= Python 3.8
- [Pybind11 catkin](https://github.com/ipab-slmc/pybind11_catkin)

## Setup / Installation

> \>= Python 3.8 is required to use the following script

To collect the required Gelsight depedencies, you can run the installation script:

```bash
./install_gelsight.sh
```

You then should be able to build as normal.

## Usage

Before running this package, you have to configure the sensor stream. For R1.5, it's recommend you use the [mjpeg_streamer](https://github.com/jacksonliam/mjpg-streamer) service on the raspberry pi and set `http_stream/url` in `gelsight.yml`.

Modify additional parameters as needed, then launch:

```bash
roslaunch gelsight_ros gelsight.launch
```

## Known Issues

If you have a 3090, you will require a specific version of PyTorch:

```
python3.8 -m pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## References

Please check the official [GelSight SDK](https://github.com/gelsightinc/gsrobotics) for more information.
