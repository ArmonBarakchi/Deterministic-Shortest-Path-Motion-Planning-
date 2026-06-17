import numpy as np


#vectorized implementation which tests a single segment AB against an
#array of obstacles
def segment_aabbs(A, B, boxes, margin: float = 0.0):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    boxes = np.asarray(boxes, dtype=float)

    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]   # single box edge case

    lo = boxes[:, :3] - margin
    hi = boxes[:, 3:] + margin
    d  = B - A

    t_enter = np.zeros(len(boxes))
    t_exit  = np.ones(len(boxes))
    alive   = np.ones(len(boxes), dtype=bool)  # boxes still in contention

    for i in range(3):
        if abs(d[i]) < 1e-12: # parallel case
            alive &= (A[i] >= lo[:, i]) & (A[i] <= hi[:, i])
        else:
            t1 = (lo[:, i] - A[i]) / d[i]
            t2 = (hi[:, i] - A[i]) / d[i]
            swap = t1 > t2
            t1[swap], t2[swap] = t2[swap].copy(), t1[swap].copy() # ensure t1 <= t2

            t_enter = np.maximum(t_enter, t1)
            t_exit  = np.minimum(t_exit,  t2)
            alive  &= t_enter <= t_exit

        if not np.any(alive):
            return False, -1

    hits = np.where(alive)[0]
    if len(hits):
        return True, int(hits[0])
    return False, -1


#check full path against every box
def path_aabbs(path, boxes, margin: float = 0.0):
    for seg_idx in range(len(path) - 1):
        hit, box_idx = segment_aabbs(path[seg_idx], path[seg_idx + 1],
                                     boxes, margin)
        if hit:
            return True, seg_idx, box_idx
    return False, -1, -1


def check_full_path(path, boundary, blocks, margin: float = 0.0):
    path = np.asarray(path, dtype=float)

    # blocks array from load_map has 9 cols and we only need the first 6
    aabbs = np.asarray(blocks, dtype=float)
    if aabbs.ndim == 1:
        aabbs = aabbs[np.newaxis, :]
    aabbs = aabbs[:, :6]

    collision, _, _ = path_aabbs(path, aabbs, margin)

    return collision

