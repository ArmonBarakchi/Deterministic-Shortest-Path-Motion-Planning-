# Motion Planning in Static and Dynamic Environments

Weighted A\*, ARA\*, RRT\*, and RRT-Connect for motion planning 3D environments, with collision testing via the slab method
-- tested across a set of obstacle environments.


## Dependency Installation
 
With pip:
 
```bash
pip install -r requirements.txt
```
 
With conda:
 
```bash
conda env create -f environment.yaml
conda activate motion-planner
```
## Structure
 
```
ECE276B_PR2/
├── starter_code/
│   ├── maps/            # Environment map files (.txt)       
│   ├── Planner2.py      # Planner class and algorithm implementations
│   ├── collision.py     # collision checking
│   ├── main.py          # Test environments and visualisation
├── environment.yaml
├── requirements.txt
└── README.md
```

## Algorithms

| Algorithm | Key Parameters |
|---|---|
| `wastar` | `epsilon`, `step`, `heuristic` |
| `arastar` | `epsilon_init`, `epsilon_final`, `epsilon_step`, `step`, `time_limit` |
| `rrtstar` | `max_iter`, `step_size`, `goal_sample_rate`, `search_radius` |
| `rrtconnect` | `max_iter`, `step_size` |

## Usage

```python
from Planner2 import MyPlanner
import numpy as np

boundary, blocks = load_map('maps/E1_flappy_bird.txt')

MP = MyPlanner(boundary, blocks, algorithm='wastar', epsilon=1.5)
path = MP.plan(start=np.array([0.5, 2.5, 5.5]),
               goal=np.array([19.0, 2.5, 5.5]))
```

To switch algorithm, change the `algorithm` argument and pass the relevant parameters:

```python
MP = MyPlanner(boundary, blocks, algorithm='arastar', epsilon_init=3.0, time_limit=5.0)
MP = MyPlanner(boundary, blocks, algorithm='rrtstar', max_iter=10000)
MP = MyPlanner(boundary, blocks, algorithm='rrtconnect', step_size=0.5)
```

## Running Tests

Uncomment the desired test at the bottom of `main.py` and run:

```bash
python main.py
```

