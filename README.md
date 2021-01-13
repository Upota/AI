# Artificial Intelligence
PACMAN Naive Q-learning Algorithms Assignments 

# Assignment 1
Implement DFS, BFS, uniformCostSearch, and A* search for a Pacman agent to find paths through his maze world, both to reach a particular location and to collect food efficiently.
* Edited files: search.py
* How to run
  <pre>
  <code>
  $ python pacman.py -l mediumMaze -p SearchAgent
  $ python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
  $ python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
  $ python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
  </code>
  </pre>  
  
# Assignment 2
Implement ReflexAgent, Minimax, and Alpha-Beta Prunung for Multi-Agent Pacman
* Edited files: multiAgents.py
* How to run
  <pre>
  <code>
  $ python autograder.py -q q1
  $ python autograder.py -q q2
  $ python autograder.py -q q3
  </code>
  </pre>
  
# Minicontest 2
Implement Q-learning, epsilon greedy, bridge crossing, approximate Q-learning
* Edited files:  {Student_ID}.py, your_baseline1.py,
your_baseline2.py, your_baseline3.py 
* How to run
  <pre>
  <code>
  $ python capture.py -r {Student_ID} -n 10
  </code>
  </pre>
  
# Final
* Edited files: qlearningAgents.py, analysis.py, featureExtractors.py, Myget_figures_ep.py, Myget_figures_alpha.py
* How to run
  <pre>
  <code>
  $ python autograder.py -q q6
  $ python autograder.py -q q7
  $ python autograder.py -q q8
  $ python autograder.py -q q9
  $ python autograder.py -q q10
  </code>
  </pre>
