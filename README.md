# Gelsight ROS

This package provides an interface for using GelSight sensors in ROS.

As of right now, only R1.5 is supported.

## Dependencies

The [GelSight SDK](https://github.com/gelsightinc/gsrobotics) is required to run this package.

## Usage

Before running this package, you have to configure the sensor stream. There are two ways you can do this:

1. Run the [mjpeg_streamer](https://github.com/jacksonliam/mjpg-streamer) service on the raspberry pi and set `cam_url` in `gelsight.yml`.
2. Install ROS on the raspberry pi and connect it to your host machine.

> The 1st recommended as support for changing raspicam parameters using OpenCV's VideoCapture class is limited. This may change in the future.

If using method 1, launch `gelsight_proc.launch`.

If using method 2, launch `gelsight.launch`.

For supported topics, read source (specifically the `gelsight_proc.py` script).
