# Goal
Implement __DFS, BFS, uniformCostSearch, and A* search__ for a Pacman agent to find paths through his maze world, both to reach a particular location and to collect food efficiently.  

## tutorial
Tutorial for getting familiar with autograder.  
Since only a few questions have been solved for each task, the results of autograd may vary.

## search
* <code>search.py</code> : Where all of my search algorithms were implemented.  
* How to run
  <pre>
  <code>
  $ python pacman.py -l mediumMaze -p SearchAgent
  $ python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
  $ python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
  $ python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
  $ python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
  </code>
  </pre>  