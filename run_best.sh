#!/bin/bash

# this script record the best parameter known for each architecture

case $1 in
    "VGG")
        python3.6 main.py --arch $1 
    ;;
    "ResNet18")
        python3.6 main.py --arch $1 --lr=0.1 --wd=5e-4 --da --nb_epochs=200
    ;;
    *)
        echo "Best param not found for this model default param will be used."
        python3.6 main.py --arch $1
    ;;
esac
