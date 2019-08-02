# mrcnn_ros

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides ros service for instance segmentation using [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). 

### Run Instruction

- Download the pre-trained weight at [here](https://drive.google.com/open?id=1g11yF89mPZKcHx3kxmIZ07XeNlixkLlz)
- Clone this repo into your catkin workspace and run `catkin_make` to build the message types
- Run the node by `python nodes/mrcnn_segment_server.py`. The config file is available [here]()
- Run `python scripts/simple_mrcnn_client_test.py` with the image in [test_data]() to test the server