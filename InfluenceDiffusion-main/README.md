
# InfluenceDiffusion

InfluenceDiffusion is a Python library that provides instruments for working with influence diffusion models on graphs. In particular, it contains implementations of
- Popular diffusion models such as Independent Cascade, (General) Linear Threshold, etc. 
- Methods for estimating parameters of these models


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install InfluenceDiffusion.

```bash
pip install InfluenceDiffusion
```

## Usage

```python
# Imports
import matplotlib.pyplot as plt
from networkx import erdos_renyi_graph

from InfluenceDiffusion.Graph import Graph # class inheriting from nx.DiGraph
from InfluenceDiffusion.influence_models import LTM 
from InfluenceDiffusion.estimation_models.EMEstimation import LTWeightEstimatorEM 
from InfluenceDiffusion.weight_samplers import make_random_weights_with_indeg_constraint

# Sample an Erdos-Renyi graph 
g_nx = erdos_renyi_graph(50, p=0.1, directed=True)
g = Graph(edge_list=g_nx.edges)

# Set ground-truth LT model edge weights (in-degree of each node is at most 1)
weights = make_random_weights_with_indeg_constraint(g, indeg_ub=1)
g.set_weights(weights)

# Sample traces from an LT model on this graph
ltm = LTM(g)
traces = ltm.sample_traces(1000)

# Estimate the weights using the traces
ltm_estimator = LTWeightEstimatorEM(g)
pred_weights = ltm_estimator.fit(traces)

# Compare with the ground-truth weights
plt.scatter(weights, pred_weights)
plt.plot([0, 1], [0, 1], linestyle='--', c='black')
plt.xlabel("True weights")
plt.ylabel("Predicted weights")
plt.show()
```

## License

MIT License

Copyright (c) 2024 Alexander Kagan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
