#!/bin/bash

# echo "load spnet checkpoints from mthrbrn"
# if [ ! -d "./checkpoint" ]; then
#   mkdir ./checkpoint
# fi
# scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/bite_selection_package/checkpoint/spnet_ckpt.pth ./checkpoint/

echo "load food dataset from mthrbrn"
scp -r conban@mthrbrn.personalrobotics.cs.washington.edu:/home/conban/conban_ws/src/conban_spanet/barnes_dataset/* .

# echo "load label_map definition from mthrbrn"
# if [ ! -d "./data" ]; then
#   mkdir ./data
# fi
# scp prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Data/foods/bite_selection_package/food_detector/food_label_map.pbtxt ./data/

