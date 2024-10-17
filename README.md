```markdown
# k-Submodular Maximization under Individual Knapsack Constraint

This repository contains the reference implementation for the paper:

**Tran, T.D., Pham, C.V., & Ha, D.T.K. (2024). k-Submodular Maximization under Individual Knapsack Constraint: Applications and Streaming.** In *IEEE Transactions on Big Data*.

The paper explores the problem of k-submodular maximization under individual knapsack constraints (kSMIK) with applications in revenue maximization and sensor placement. It introduces a novel streaming algorithm that achieves a constant approximation ratio while operating with minimal query complexity and space consumption. 

Full version of the paper is available [here](https://doi.org/10.1145/3628797.3628843).

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
