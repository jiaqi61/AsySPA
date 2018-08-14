# AsySPA
An exact asynchronous algorithm for distributed optimization over digraphs based on [this paper](https://arxiv.org/abs/1808.04118).

The codes are tested on Linux (Ubuntu 16.04) with OpenMPI and Python 3.6. 

# Examples

- To run a demo on a least-square problem, execute the following command

```bash
mpiexec -np $(num_nodes) -bind-to core -allow-run-as-root python ./asyspa/asy_gradient_push.py
```
The `-bind-to core` is optional, which let the MPI bind processes to cores, and hence the  `num_nodes`  should not be larger then the cores in your machine.

The command above will distributedly solve the following problem using the AsySPA

![tex](https://latex.codecogs.com/png.download?%5Ctext%7Bminimize%7D_%7Bx%5Cin%5Cmathbb%7BR%7D%5E%7B10%7D%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bnum%5C_nodes%7D%5C%7CA_i%5E%7B200%5Ctimes10%7Dx-b_i%5E%7B200%7D%5C%7C%5E2) 

where the data is randomly generated.

- To distributedly train a multi-class logistic regression classifier on the [Covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype) as in the paper,  run the following command first

```bash
python ./asyspa/preprocess_dataset_covtype.py --n $(num_nodes) --file ./dataset_covtype/covtype.csv
```

This will preprocess the data and divide it into `num_nodes`  parts. Then, run the following command to start distributed training

```bash
mpiexec -np $(num_nodes) -bind-to core -allow-run-as-root python ./asyspa/distributed_asy_logistic_regression.py --data_dir ./dataset_covtype/data_partition_$(num_nodes) --save_dir ./result/core_$(num_nodes)
```

This will train the classifier for 300 seconds in default, and you can change the time by passing some arguments. See the file for details.