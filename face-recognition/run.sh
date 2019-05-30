#!/bin/bash

cd /home/pi/home-automation/raspi/face-recognition/ && python3 -u recognize.py
trap 'exit 0' INT
exit 0

