# Pendulum
Feed Forward Network to learn the Pendulum-v0 dynamics and a graph planner algorithm to solve it

To generate the data for training the neural network, use the command "python generate_data.py". This code runs multiple episodes of simulation by randomly sampling actions at every step to generate data. 

To split the data into smaller sets, run split_data.py. This file splits the data into multiple, equally sized, disjoint subsets. The "generate_data.py" file generates 5 million examples and hence, using a subset of this dataset by selecting one or more of the splits helps train the model faster. 

All the data generated above is stored in the "data/" folder.

To train the model, run "train.sh". It essentially runs "train_nn.py" with a bunch of command line arguments. Note that you need to provide the path of the folder where you want to save the model and it will be saved as "model.hdf5" in that folder. 

To run the graph planner algorithm (A*), run "run_graph_planner.sh". It essentially runs the file "graph_planner.py" with a bunch of command line arguments. You need to provide the complete path of the model to be used for building the graph. 


The command line arguments and their format for both the files "train_nn.py" and "graph_planner.py" can be seen in the respecticve bash files.
