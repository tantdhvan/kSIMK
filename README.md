# k-Submodular Maximization under Individual Knapsack Constraint: Applications and Streaming

This repository contains the reference implementation for the paper:

**Tan D. Tran, Canh V. Pham, & Dung T.K. Ha, k-Submodular Maximization under Individual Knapsack Constraint: Applications and Streaming.**

The $k$-submodular optimization problems have been an active research area with many applications in machine learning, such as data summarization, influence maximization, and maximum revenue. 
    In this work, we investigate the problem of $k$-submodular maximization under individual knapsack constraints (**kSMIK**) that capture various realities of such applications.
    We solve the problem in the streaming setting that requires two passes over the data and a reasonable amount of memory. In particular,
    we propose a streaming algorithm named **Str**, which runs in **$O(nk\log (B)/\epsilon)$** query complexity, takes **$O(B\log (n)/\epsilon)$** space complexity, and achieves an approximation ratio of **$\frac{1-\epsilon}{2(k+1)}$** for monotone objective function; **$\frac{1-\epsilon}{2k+3}$** for the non-monotone objective function, where $n$ is the size of the ground set, $B$ is the total budget and $\epsilon\in (0,1)$ is an input parameter.
    We validate our algorithm in two applications of the studied problem, including revenue maximization and sensor placement on some benchmark datasets. The experiment results show that our streaming algorithm provides higher-quality solutions with lower query complexity than the existing based-line algorithm.

## Algorithms Included

- **StrOpt**: A streaming algorithm with known optimal values.
- **Str**: The primary streaming algorithm that does not assume knowledge of the optimal values.
  
Both algorithms are tested and evaluated through experiments on real-world datasets.

### Dependencies

- GCC (g++), GNU Make
- Python 3 (with `matplotlib` for plotting results)


### Running the Algorithms

**Đối với ứng dụng $k$-type Product Revenue Maximization Under Individual Knapsack Constraint ($\kPMIK$):**
```bash
cd revenue_max
make
./revmax -g <input file> -b <budget> -G -e <epsilon>
```
**Example:**
```bash
./revmax -g data/astro.bin -b 0.01 -q -G -e 0.1
./revmax -g data/astro.bin -b 0.01 -q -Q -e 0.1
```

### Results

The paper includes experiments with the following datasets:
- Facebook social network
- Astro Physics collaboration network
- Enron email network
- Intel Lab sensor data

The output consists of performance metrics such as the objective function values, number of queries, and running time.

### License

This code is licensed under the MIT License.
```
