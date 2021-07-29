# Goal
In this project, the ghosts walk around to catch Pacman.  
The goal is to move as Pacman intended, not win the game.  
Implement __ReflexAgent, Minimax, and Alpha-Beta Prunung__ for Multi-Agent Pacman.  

## multiagent
* <code>multiAgents.py</code> : Where all of my acting rule algorithms were implemented.  
* How to run
  <pre>
  <code>
  $ python pacman.py -p ReflexAgent -l testClassic
  $ python autograder.py -q q1

  $ python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
  $ python autograder.py -q q2

  $ python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
  $ python autograder.py -q q3
  </code>
</pre>  