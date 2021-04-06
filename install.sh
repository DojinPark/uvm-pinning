#!/bin/bash
sudo systemctl isolate multi-user.target
sudo $DRIVER_PATH/nvidia-installer
sudo systemctl start graphical.target
