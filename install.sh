#!/bin/bash
sudo systemctl isolate multi-user.target
sudo chmod +x $DRIVER_PATH/
sudo $DRIVER_PATH/nvidia-installer
sudo systemctl start graphical.target