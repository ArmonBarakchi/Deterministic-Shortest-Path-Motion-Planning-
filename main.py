import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collision import check_full_path
from Planner2 import MyPlanner


def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  goal = np.atleast_2d(goal)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[:,0],goal[:,1],goal[:,2],'go',markersize=7,markeredgecolor='k')  
  for i, (x, y, z) in enumerate(goal):
    ax.text(x+0.7, y+0.2, z, str(i + 1), fontsize=10, color='k')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def animate_dynamic_target(ax, goal, paths, times, goal1_stay, goal_other_stay, slowdown=5):
  goal_stays = [goal1_stay] + [goal_other_stay] * (goal.shape[0] - 1)
  goal_event_times = np.cumsum(goal_stays)
  path_event_times = np.cumsum(times)

  goal_idx = 0
  path_idx = 0
  current_goal_handle = ax.plot(
      [goal[0,0]], [goal[0,1]], [goal[0,2]],
      'bo', markersize=9, markeredgecolor='k'
  )
  current_path_handle = None
  start_time = time.time()

  while goal_idx < len(goal) or path_idx < len(paths):
    sim_time = (time.time() - start_time) / slowdown

    if goal_idx < len(goal) and sim_time >= goal_event_times[goal_idx]:
      if current_goal_handle is not None:
        current_goal_handle[0].remove()
      if goal_idx < len(goal) - 1:
        current_goal = goal[goal_idx + 1]
        current_goal_handle = ax.plot(
            [current_goal[0]], [current_goal[1]], [current_goal[2]],
            'bo', markersize=9, markeredgecolor='k'
        )
      goal_idx += 1

    if path_idx < len(paths) and sim_time >= path_event_times[path_idx]:
      if current_path_handle is not None:
        current_path_handle[0].remove()
      path = paths[path_idx]
      current_path_handle = ax.plot(
          path[:, 0], path[:, 1], path[:, 2],
          'r-', linewidth=2
      )
      path_idx += 1

    plt.draw()
    plt.pause(0.01)

  if current_path_handle is not None:
    current_path_handle[0].remove()
  return path_event_times <= goal_event_times


def runtest(mapfile, start, goal, verbose = True, dynamic_target = False):
  '''
  This function:
   * loads the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''

  # Load a map and instantiate a motion planner
  boundary, blocks = load_map(mapfile)


  # Note: uncomment the chose planner for each environment

  #MP = MyPlanner(boundary, blocks, algorithm='wastar', epsilon=1.5)
  #MP = MyPlanner(boundary, blocks, algorithm='rrtconnect', step_size=0.5)
  MP = MyPlanner(boundary, blocks, algorithm='arastar', epsilon_init=3.0, time_limit=5)
  #MP = MyPlanner(boundary, blocks, algorithm='rrtstar')


  # Display the environment
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

  # Call the motion planner
  if dynamic_target:
    paths = []
    times = []
    for i in range(goal.shape[0]):
      t0 = tic()
      path = MP.plan(start, goal[i])
      elapsed = time.time() - t0
      toc(t0,"Planning")
      times.append(elapsed)
      paths.append(path)  
  else:
      t0 = tic()
      path = MP.plan(start, goal)
      toc(t0,"Planning")

  # Visualize the dynamic moving process and check if the goal is reached in required time
  if dynamic_target:
    mapfile_lower = mapfile.lower()
    if 'window' in mapfile_lower:
      goal1_stay = 1.5
      goal_other_stay = 1
    else:
      goal1_stay = 0.3
      goal_other_stay = 0.1
    slowdown = 0.1 # Time slowdown factor for visualization, tune this value to slow down the visualization
    path_success = animate_dynamic_target(
        ax, goal, paths, times, goal1_stay, goal_other_stay, slowdown
    )

  # Plot the path
  if verbose:
    if dynamic_target:
      for i, path in enumerate(paths):
        if path_success[i]:
          ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')
        else:
          ax.plot(
            [goal[i, 0]], [goal[i, 1]], [goal[i, 2]],
            'mo', markersize=9, markeredgecolor='k'
          )
    else:
      ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')


  # TODO: You should verify whether all paths actually intersects any of the obstacles in continuous space
  # TODO: You can implement your own algorithm or use an existing library for segment and 
  #       axis-aligned bounding box (AABB) intersection

  collision = check_full_path(path, boundary, blocks) #redundancy to make sure that the _Env class works properly
  if dynamic_target:
    goal_reached_list = []
    pathlength = []
    for i, path in enumerate(paths):
        reached = np.sum((path[-1] - goal[i])**2) <= 0.1
        goal_reached_list.append(reached)

        pathlength.append(np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))))
    goal_reached = np.all(goal_reached_list)
    success = (not collision) and goal_reached and np.all(path_success)
    total = 0
    for item in pathlength:
      total +=item
    print(f"Total path length: {total}")
  else:
    goal_reached = np.sum((path[-1] - goal)**2) <= 0.1
    pathlength = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    success = (not collision) and goal_reached 
  return success, pathlength


def test_single_cube(verbose = True, dynamic_target = False):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  success, pathlength = runtest('./maps/E4_single_cube.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')
  
  
def test_maze(verbose = True, dynamic_target = False):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, pathlength = runtest('./maps/E2_maze.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

    
def test_window(verbose = True, dynamic_target = True):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([
    [8.800, 12.300, 3.800],
    [7.687, 13.227, 4.449],
    [5.000, 13.610, 4.718],
    [2.313, 13.227, 4.449],
    [1.200, 12.300, 3.800],
    [2.313, 11.373, 3.151],
    [5.000, 10.990, 2.882],
    [7.687, 11.373, 3.151],
  ])
  success, pathlength = runtest('./maps/E6_window.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %s'%pathlength)
  print('\n')

  
def test_tower(verbose = True, dynamic_target = False):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, pathlength = runtest('./maps/E5_tower.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

     
def test_flappy_bird(verbose = True, dynamic_target = False):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  success, pathlength = runtest('./maps/E1_flappy_bird.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength) 
  print('\n')

  
def test_room(verbose = True, dynamic_target = True):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([[1.7, 0.5 , 1.7],
           [8.0, 1.0, 1.5],
           [6.0, 4.0, 3.0],
           [3.0, 3.6, 0.5],
           [3.0, 7.0 , 1.0],
           [6.0, 8.0, 0.5],
           [8.0, 6.0 , 1.5],
           [9, 7.5, 0.5]])
  success, pathlength = runtest('./maps/E7_room.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %s'%pathlength)
  print('\n')


def test_monza(verbose = True, dynamic_target = False):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  success, pathlength = runtest('./maps/E3_monza.txt', start, goal, verbose, dynamic_target)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


if __name__=="__main__":
  # #Static target tests
  test_single_cube()
  test_maze()
  test_flappy_bird()
  test_monza()
  test_tower()

  # #Dynamic target tests
  #test_window()
  #test_room()
  plt.show(block=True)








