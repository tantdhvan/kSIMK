# k-Submodular Maximization under Individual Knapsack Constraint: Applications and Streaming.

This repository contains the reference implementation for the paper:

**Tan D. Tran, Canh V. Pham, & Dung T.K. Ha. k-Submodular Maximization under Individual Knapsack Constraint: Applications and Streaming.**

The paper explores the problem of k-submodular maximization under individual knapsack constraints (kSMIK) with applications in revenue maximization and sensor placement. It introduces a novel streaming algorithm that achieves a constant approximation ratio while operating with minimal query complexity and space consumption. 

## Algorithms Included

- **Str**: The streaming algorithm that does not assume knowledge of the optimal values.
- **Greedy**: The greedy algorithm.
  
Both algorithms are tested and evaluated through experiments on real-world datasets.

### Dependencies

- GCC (g++), GNU Make

### Binaries

The following binaries are included:

- `revenue_max`: Executes algorithms on revenue maximization problem.
- `sensor_placement`: Executes algorithms on the sensor placement problem.

### Input Format

Input is provided as an undirected edge list for the revenue maximization problem or sensor placement data for sensor placement. The input for edge lists is formatted as:

```
<From id> <To id> <Weight (if applicable)>
```

Node IDs should be non-negative integers. Weights are unsigned integers where applicable.

### Running the Algorithms

1. **Revenue Maximization**: Run the following command to execute algorithms on the revenue maximization problem:

```bash
./revenue_max -g <input file> -k <number of products> -B <budget per product> -e <epsilon> -q
```

2. **Sensor Placement**: Run the following command to execute algorithms on the sensor placement problem:

```bash
./sensor_placement -g <input file> -k <number of sensors> -B <budget per sensor> -e <epsilon> -q
```

### Parameters

```
-g <input file>             # Input file in edge list format
-k <number>                 # Number of products (for revenue maximization) or sensors (for sensor placement)
-B <budget per category>    # Budget constraint
-e <epsilon>                # Approximation parameter
-q                          # Quiet mode
```

### Example

To reproduce the experiments from the paper, run:

```bash
make reproduce
```

This will execute experiments on the `Facebook`, `Astro`, and `Enron` datasets for revenue maximization and the `Intel Lab` dataset for sensor placement. The results will be generated and saved as PDF files.

### Results

The paper includes experiments with the following datasets:
- Facebook social network
- Astro Physics collaboration network
- Enron email network
- Intel Lab sensor data

The output consists of performance metrics such as the objective function values, number of queries, and running time.

### Reproduce Experiments

To reproduce the experiments as presented in the paper, run:

```bash
make reproduce
```

This command will run the experiments and generate the results, including plots, in the corresponding subdirectories.

### License

This code is licensed under the MIT License.
```
