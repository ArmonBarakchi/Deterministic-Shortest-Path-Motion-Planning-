import numpy as np
import heapq
import math
import time
import random
from collision import segment_aabbs


#This class emulates the environment, but puts a thin wrapper around it to stop the planner from
#touching the boundary, additionally it gives the free space by using the collision checker
class _Env:
    def __init__(self, boundary, blocks):
        self.boundary = np.asarray(boundary, dtype=float)
        b = np.asarray(blocks, dtype=float)
        if b.ndim == 1:
            b = b[np.newaxis, :]
        self.aabbs = b[:, :6]
        self.lo    = self.boundary[0, :3].copy()
        self.hi    = self.boundary[0, 3:6].copy()

    def in_bounds(self, p):
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def point_free(self, p):
        if not self.in_bounds(p):
            return False
        hit, _ = segment_aabbs(p, p, self.aabbs)
        return not hit

    def segment_free(self, A, B):
        if not (self.in_bounds(A) and self.in_bounds(B)):
            return False
        hit, _ = segment_aabbs(A, B, self.aabbs)
        return not hit

    def random_free_point(self):
        for _ in range(1000):
            p = self.lo + np.random.rand(3) * (self.hi - self.lo)
            if self.point_free(p):
                return p
        return None



#  Grid construction helpers shared by A*-family algorithms

_OFFSETS = []
_COSTS   = {}
for _dx in (-1, 0, 1):
    for _dy in (-1, 0, 1):
        for _dz in (-1, 0, 1):
            if _dx == _dy == _dz == 0:
                continue
            o = (_dx, _dy, _dz)
            _OFFSETS.append(o)
            _COSTS[o] = math.sqrt(_dx*_dx + _dy*_dy + _dz*_dz)


def _to_grid(p, step):
    return tuple(np.round(np.asarray(p) / step).astype(int))

def _to_world(g, step):
    return np.array(g, dtype=float) * step

def _reconstruct(parent, node, step, goal):
    path = []
    cur = node
    while cur is not None:
        path.append(_to_world(cur, step))
        cur = parent[cur]
    path.reverse()
    path.append(np.asarray(goal, dtype=float))
    return np.array(path)


#weighted A*. Supports heuristic changes, just change the default argument here.
def _wastar(start, goal, env, epsilon=1.5, step=0.5, heuristic='euclidean'):
    sg = _to_grid(start, step)

    def h(g):
        w = _to_world(g, step)
        dx = abs(w[0] - goal[0])
        dy = abs(w[1] - goal[1])
        dz = abs(w[2] - goal[2])

        if heuristic == 'euclidean':
            val = math.sqrt(dx**2 + dy**2 + dz**2)
        elif heuristic == 'manhattan':
            val = dx + dy + dz
        elif heuristic == 'octile':
            d1, d2, d3 = sorted([dx, dy, dz], reverse=True)
            val = d3 * math.sqrt(3) + (d2 - d3) * math.sqrt(2) + (d1 - d2)
        return epsilon * val


    g_score = {sg: 0.0}
    parent  = {sg: None}
    closed  = set()
    explored = 0
    OPEN    = [(h(sg), 0.0, sg)]

    while OPEN:
        _, g, curr = heapq.heappop(OPEN)
        if curr in closed:
            continue
        closed.add(curr)
        explored +=1

        # close enough to goal
        if np.linalg.norm(_to_world(curr, step) - goal) <= step * 1.5:
            print(f"number of nodes explored: {explored}")
            return _reconstruct(parent, curr, step, goal)

        curr_w = _to_world(curr, step)
        for off in _OFFSETS:
            neighbor = (curr[0]+off[0], curr[1]+off[1], curr[2]+off[2])

            if neighbor in closed:
                continue
            neighbor_w = _to_world(neighbor, step)

            if not env.segment_free(curr_w, neighbor_w):
                continue

            # label correcting update
            tg = g + _COSTS[off] * step
            if tg < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tg
                parent[neighbor]  = curr
                heapq.heappush(OPEN, (tg + h(neighbor), tg, neighbor))

    print(f"number of nodes explored: {explored}")
    return np.array([start, goal])


#weighted A*. Supports heuristic changes, just change the default argument here.

def _arastar(start, goal, env, epsilon_init=3.0, epsilon_final=1.0,
             epsilon_step=0.5, step=0.5, time_limit=5.0, heuristic='euclidean'):

    sg = _to_grid(start, step)

    def hval(g, eps):
        w = _to_world(g, step)
        dx = abs(w[0] - goal[0])
        dy = abs(w[1] - goal[1])
        dz = abs(w[2] - goal[2])

        if heuristic == 'euclidean':
            val = math.sqrt(dx**2 + dy**2 + dz**2)
        elif heuristic == 'manhattan':
            val = dx + dy + dz
        elif heuristic == 'octile':
            d1, d2, d3 = sorted([dx, dy, dz], reverse=True)
            val = d3 * math.sqrt(3) + (d2 - d3) * math.sqrt(2) + (d1 - d2)
        return eps * val

    g_score = {sg: 0.0}
    parent = {sg: None}
    best_path = np.array([start, goal])
    t0 = time.time()
    total_expanded = 0

    eps = epsilon_init
    while eps >= epsilon_final - 1e-9:
        closed = set()
        iter_expanded = 0
        OPEN = [(g_score.get(sg, 0.0) + hval(sg, eps), 0.0, sg)]

        while OPEN:
            if time.time() - t0 > time_limit:
                print(f"timeout at eps={eps}, "
                      f"iter_expanded={iter_expanded}, total={total_expanded}")
                return best_path

            _, g, curr = heapq.heappop(OPEN)
            if curr in closed:
                continue
            closed.add(curr)
            iter_expanded += 1
            total_expanded += 1

            if np.linalg.norm(_to_world(curr, step) - goal) <= step * 1.5:
                best_path = _reconstruct(parent, curr, step, goal)
                print(f"[arastar] eps={eps}  "
                      f"iter_expanded={iter_expanded}  "
                      f"total={total_expanded}  "
                      f"path_len={np.sum(np.sqrt(np.sum(np.diff(best_path, axis=0) ** 2, axis=1)))}")
                break

            curr_w = _to_world(curr, step)
            for off in _OFFSETS:
                neighbor = (curr[0]+off[0], curr[1]+off[1], curr[2]+off[2])
                if neighbor in closed:
                    continue
                neighbor_w = _to_world(neighbor, step)
                if not env.segment_free(curr_w, neighbor_w):
                    continue
                tg = g + _COSTS[off] * step
                if tg < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tg
                    parent[neighbor]  = curr
                    heapq.heappush(OPEN, (tg + hval(neighbor, eps), tg, neighbor))

        eps = round(eps - epsilon_step, 10)

    return best_path


#RRT* algorithm
def _rrtstar(start, goal, env, max_iter=5000, step_size=0.5,
             goal_sample_rate=0.1, search_radius=1.5):

    nodes  = [np.asarray(start, dtype=float)]
    parent = {0: None}
    cost   = {0: 0.0}
    goal   = np.asarray(goal, dtype=float)

    #helpers
    def nearest(p):
        return int(np.argmin([np.linalg.norm(n - p) for n in nodes]))

    def steer(frm, to):
        d = np.linalg.norm(to - frm)
        if d < 1e-9:
            return frm.copy()
        return frm + min(step_size, d) / d * (to - frm)

    def near(p, r):
        return [i for i, n in enumerate(nodes) if np.linalg.norm(n - p) <= r]

    best_goal_idx  = None
    best_goal_cost = math.inf
    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        sample = goal.copy() if random.random() < goal_sample_rate \
                 else env.random_free_point()
        if sample is None:
            continue

        nidx  = nearest(sample)
        new_p = steer(nodes[nidx], sample)
        if not env.segment_free(nodes[nidx], new_p):
            continue

        # choose best parent among neighbours
        near_idxs   = near(new_p, search_radius)
        best_parent = nidx
        best_cost   = cost[nidx] + np.linalg.norm(new_p - nodes[nidx])

        for i in near_idxs:
            c = cost[i] + np.linalg.norm(new_p - nodes[i])
            if c < best_cost and env.segment_free(nodes[i], new_p):
                best_cost, best_parent = c, i

        new_idx = len(nodes)
        nodes.append(new_p)
        parent[new_idx] = best_parent
        cost[new_idx]   = best_cost

        # rewire: update neighbours that benefit from routing through new_p
        for i in near_idxs:
            c = best_cost + np.linalg.norm(nodes[i] - new_p)
            if c < cost[i] and env.segment_free(new_p, nodes[i]):
                parent[i] = new_idx
                cost[i]   = c

        # check goal
        if np.linalg.norm(new_p - goal) <= step_size:
            total = best_cost + np.linalg.norm(new_p - goal)
            if total < best_goal_cost and env.segment_free(new_p, goal):
                best_goal_cost = total
                best_goal_idx  = new_idx

    if best_goal_idx is None:
        print(iter_count)
        return np.array([start, goal])

    path = [goal]
    idx  = best_goal_idx
    while idx is not None:
        path.append(nodes[idx])
        idx = parent[idx]
    path.reverse()
    print(iter_count)
    return np.array(path)


#RRT-connect algorithm
def _rrtconnect(start, goal, env, max_iter=5000, step_size=0.5):

    start = np.asarray(start, dtype=float)
    goal  = np.asarray(goal,  dtype=float)

    Ta_nodes  = [start.copy()];  Ta_parent = {0: None}
    Tb_nodes  = [goal.copy()];   Tb_parent = {0: None}


    #helpers

    def nearest(tree, p):
        return int(np.argmin([np.linalg.norm(n - p) for n in tree]))

    def steer(frm, to):
        d = np.linalg.norm(to - frm)
        if d < 1e-9:
            return frm.copy()
        return frm + min(step_size, d) / d * (to - frm)

    TRAPPED, ADVANCED, REACHED = 0, 1, 2

    def extend(tree, pmap, p):
        nidx  = nearest(tree, p)
        new_p = steer(tree[nidx], p)
        if env.segment_free(tree[nidx], new_p):
            new_idx = len(tree)
            tree.append(new_p)
            pmap[new_idx] = nidx
            status = REACHED if np.linalg.norm(new_p - p) < 1e-6 else ADVANCED
            return status, new_idx
        return TRAPPED, -1

    def connect(tree, pmap, p):
        status, last = ADVANCED, -1
        while status == ADVANCED:
            status, last = extend(tree, pmap, p)
        return status, last

    def extract(tree, pmap, leaf):
        path = []
        idx  = leaf
        while idx is not None:
            path.append(tree[idx])
            idx = pmap[idx]
        path.reverse()
        return path

    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        sample = env.random_free_point()
        if sample is None:
            continue

        status_a, idx_a = extend(Ta_nodes, Ta_parent, sample)
        if status_a != TRAPPED:
            status_b, idx_b = connect(Tb_nodes, Tb_parent, Ta_nodes[idx_a])
            if status_b == REACHED:
                path_a = extract(Ta_nodes, Ta_parent, idx_a)
                path_b = extract(Tb_nodes, Tb_parent, idx_b)
                path_b.reverse()
                print(iter_count)
                return np.array(path_a + path_b)

        # swap so both trees grow equally
        Ta_nodes,  Tb_nodes  = Tb_nodes,  Ta_nodes
        Ta_parent, Tb_parent = Tb_parent, Ta_parent
    print(iter_count)
    return np.array([start, goal])


#planner class
_ALGORITHMS = {
    'wastar':     _wastar,
    'arastar':    _arastar,
    'rrtstar':    _rrtstar,
    'rrtconnect': _rrtconnect,
}

#I know its lazy but I just used this to change the parameters used
_DEFAULTS = {
    'wastar':     {'epsilon': 1.5,  'step': 0.5},
    'arastar':    {'epsilon_init': 3.0, 'epsilon_final': 1.0,
                   'epsilon_step': 0.5, 'step': 0.5, 'time_limit': 5.0},
    'rrtstar':    {'max_iter': 10000, 'step_size': 0.25,
                   'goal_sample_rate': 0.2, 'search_radius': .1},
    'rrtconnect': {'max_iter': 3000, 'step_size': 0.5},
}


class MyPlanner:

    def __init__(self, boundary, blocks, algorithm='wastar', **kwargs):
        algorithm = algorithm.lower()
        self._env     = _Env(boundary, blocks)
        self._algo_fn = _ALGORITHMS[algorithm]
        self.algorithm = algorithm
        params = dict(_DEFAULTS[algorithm])
        params.update(kwargs)
        self._kwargs = params

    def plan(self, start, goal):
        start = np.asarray(start, dtype=float)
        goal  = np.asarray(goal,  dtype=float)
        return self._algo_fn(start, goal, self._env, **self._kwargs)