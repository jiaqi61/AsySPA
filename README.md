# AsySPA
An exact asynchronous algorithm for distributed optimization over digraphs based on [this paper](https://arxiv.org/abs/1808.04118).

The codes are tested on Linux (Ubuntu 16.04) with OpenMPI and Python 3.6. 

# Examples

- To run a demo on a least-square problem, execute the following command

```bash
mpiexec -np $(num_nodes) -bind-to core -allow-run-as-root python ./asyspa/asy_gradient_push.py
```
The `-bind-to core` is optional, which let the MPI binds each process to a core, and hence the  `num_nodes`  should not be larger than the number of cores in your machine.

The command above will distributedly solve the following problem using the AsySPA

![tex](http://latex.codecogs.com/gif.latex?\\text{minimize}_{x\\in\\mathbb{R}^{10}} \\sum_{i=1}^{num\\_nodes}\\|A_i^{200\\times10}x-b_i^{200}\\|^2) 

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