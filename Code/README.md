## Note

As Ising models and Quadratic Unconstrained Binary Optimization (QUBO) models are mathematically equal<cite>[1]</cite><cite>[2]</cite>. For the convenience of coding, we convert the Ising model into a QUBO model in our code to do mathematical calculation with equation

$$x_i=\frac{1-z_i}{2}$$

The new converted QUBO model in the code and the fitness values achieved are mathematically equal to the Ising model in the paper.

According to the QUBO model formulation, in our results, we use binary numbers to represent the selection of test cases. If a test case is selected, we represent it with 1. Otherwise, we represent it with 0. 


[1]: Glover, Fred, Gary Kochenberger, and Yu Du. "A tutorial on formulating and using QUBO models." arXiv preprint arXiv:1811.11538 (2018).
[2]: Lucas, Andrew. "Ising formulations of many NP problems." Frontiers in physics 2 (2014): 5.


