#!/bin/bash

# this script record the best parameter known for each architecture

case $1 in

    "ConvNet")
        python3.6 main.py --arch $1 --on_graph --lr=0.02 --wd=5e-4 --nb_epochs=150 --ls 50 100 150  
    ;;
    *)
        echo "Best param not found for this model default param will be used."
        python3.6 main.py --arch $1 --on_graph
    ;;
esac
