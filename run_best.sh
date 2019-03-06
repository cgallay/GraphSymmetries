#!/bin/bash

# this script record the best parameter known for each architecture

case $1 in
"VGG")
python3.6 main.py --arch $1 
;;
"ResNet")
python3.6 main.py --arch $1
;;
*)
echo "Model not found"
esac


