# Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search

This is a tensorflow implementation of solving the maximum indepedent set problem using graph convolutional networks and guided tree search. The graph convolutional networks implementation is based on [GCN](https://github.com/tkipf/gcn) (MIT License).

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.3) + Scipy + Numpy.

Tested in Ubuntu 16.04 LTS + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=6.0). CPU mode should also work with no/minor changes.

### Quick Start (Testing)
* Note the result produced here is without graph reduction and local search. If you wish to use them, please see the instructions in the next subsection.
1. Clone this repository.
2. Run "python demo.py" or "python demo_parallel.py" to solve the problem instance(s) in the "data" folder.
3. The result will be saved in "res_600".

### Instructions for using graph reduction and local search
1. Clone KaMIS (GPLv2 License) from its [Project](http://algo2.iti.kit.edu/kamis/) or [GitHub](https://github.com/sebalamm/KaMIS/) page.
2. Copy the files in "kernel" of this repo to "KaMIS" and run make. This will generate a shared object file "libreduce.so".
3. Copy "libreduce.so" back to "kernel".
4. Uncomment lines 8, 20, 87, 109, 272, 300 and comment 88, 110, 273 in "demo_parallel.py" to enhance it with graph reduction and local search. "demo.py" can be modified accordingly to include these components.
5. Rerun "demo_parallel.py" to see the difference.

### Training
Run "train.py" to start training. Note that the training data is not provided and should be downloaded or synthesized from other sources, e.g.,

SATLIB (https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)

RB Model (https://github.com/notbad/RB)

The data file should contain an adjacency matrix and at least one groundtruth label, and saved in a MATLAB file with "adj" and "indset_label" symbols. The default data path is defined as data_path = "./data/CBS_Graph".

## Citation
If you use our code for research, please cite our paper:

Zhuwen Li, Qifeng Chen and Vladlen Koltun. Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search. In NIPS 2018.

## Todo List
1. Implement graph reduction and local search, and release them under MIT License

## Question
If you have any question or request about the code and data, please email me at lzhuwen@gmail.com.

## License
MIT License
