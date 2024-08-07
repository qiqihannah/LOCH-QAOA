# Test Case Optimization with a Quantum Approximate Optimization Algorithm

## Abstract
Test case optimization (TCO) reduces the software testing cost while preserving its effectiveness. However, to solve TCO problems for large-scale and complex software systems, substantial computational resources are required. Quantum approximate optimization algorithms (QAOAs) are promising combinatorial optimization algorithms that rely on quantum computational resources, with the potential to offer increased efficiency compared to classical approaches. Several proof-of-concept applications of QAOAs for solving combinatorial problems, such as portfolio optimization, energy optimization in power systems, and job scheduling, have been proposed. Given the lack of investigation into QAOA's application for TCO problems, and motivated by the computational challenges of TCO problems and the potential of QAOAs, we present IGDec-QAOA to formulate a TCO problem as a QAOA problem and solve it on both ideal and noisy quantum computer simulators, as well as on a real quantum computer. To solve bigger TCO problems that require many qubits, which are unavailable these days, we integrate a problem decomposition strategy with the QAOA. We performed an empirical evaluation with five TCO problems and four publicly available industrial datasets from ABB, Google, and Orona to compare various configurations of IGDec-QAOA, assess its decomposition strategy of handling large datasets, and compare its performance with classical algorithms (i.e., Genetic Algorithm (GA) and Random Search). Based on the evaluation results achieved on an ideal simulator, we recommend the best configuration of our approach for TCO problems. Also, we demonstrate that our approach can reach the same effectiveness as GA and outperform GA in two out of five test case optimization problems we conducted. In addition, we observe that, on the noisy simulator, IGDec-QAOA achieved similar performance to that from the ideal simulator. Finally, we also demonstrate the feasibility of IGDec-QAOA on a real quantum computer in the presence of noise.

## Overview
<img src="image/overview.png" width="800">

This repository contains four industrial datasets and the code for producing results of running LOCH-QAOA and baseline approaches for TCO problems.

It also contains the original results reported in the paper and the analyzing code.

## Structure of the repository

* Folder *Data* contains the information of 4 industrial datasets, and a small dataset extracted from Paint Control dataset , including all the test cases and their attribute values.
* Folder *Code* contains the code of LOCH-QAOA, Div-QAOA, and two classical algorithms, random search and genetic algorithm. It also contains a Jupyter Notebook of running LOCH-QAOA with the Qiskit quantum computer on dataset Paintcontrol.
* Folder *Analyse* contains the code of analysing the experiments and results.
* Experiment results in the paper are saved in [this link](https://drive.google.com/drive/folders/1zvqdwVx5RZeVq1ljI7EoywKN5bWV9BI7?usp=drive_link).
    * subfolder *LOCH-QAOA-result*: each folder under it is named by qaoa_p (i.e., layers of QAOA). which contains results of 5 case studies stored in .zip files. 
      * The .zip files contain experiment results of all 30 runs of LOCH-QAOA with different sizes. In each run, we report results of each iteration, log information of each sub-problem and the final solution.
    * subfolder *hardware-result*: it contains the results of the Paint Control case study.
      * The subfolders contain the experiment results of 5 iterations and the information of each job. 
    * subfolder *noisy_simulator-result*: it contains the results of all 30 runs on 5 datasets.
      * The subfolders contain the experiment results of all 30 runs. We report the best recorded *fval* values achieved in each iteration in that run.
    * subfolder *Div-QAOA-result*: it contains experiment results of all 5 case studies. 
      * The subfolders under contain experiment results of all 30 runs. We report the information of each sub-problem and their optimal solution.
    * subfolder *GA-result*: it contains the experiment results of all 5 case studies. 
      * dataset_log folder contains the evolution results of each generation with various population sizes (i.e., 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) and their corresponding optimal fitness values, execution time, and number of evaluations of each run (in gen_file_popsize.csv).
      * dataset.csv file contains the average results of various population sizes.
      * selected.csv file contains the average results with the optimal population sizes of each case study.
    * subfolder *RandomSearch-result*: it contains the results of all 5 case studies.
      * The subfolders contain the experiment results of all 30 runs. We report the best recorded *fval* values achieved in each iteration in that run.


## Installation of running the code
We use [Qiskit](https://qiskit.org/) as a quantum framework. We use jMetalPy to implement the classical search algorithms (i.e., GA and RandomSearch)

The following steps should guarantee to have a working installation:
* Install Anaconda. You can download Anaconda for your OS from [https://www.anaconda.com/](https://www.anaconda.com/)
* Create an environment (e.g., with name "quantum_env")
    * conda create -n quantum_env python=3.9
* Activate the environment, and install qiskit, qiskit optimization, jMetalPy, docplex
    * conda activate quantum_env
    * pip install qiskit
    * pip install qiskit[optimization]
    
## Running our code

The approach can be run as follows.

First, you need to activate the conda environment:

```
conda activate quantum_env
```

In order to run LOCH-QAOA, you can run the following code:

```
python Code/loch-qaoa/loch_qaoa_tcm.py [run] [size] [p] [file name]
```

For example, the 1st run of our approach with QAOA sub-problem size 7 and 1 layer for dataset Paint Control:

```
python Code/loch-qaoa/loch_qaoa_tcm.py 1 7 1 paintcontrol
```

In order to run two case studies related to ELEVATOR, we can run the file loch_qaoa_elev_three.py and loch_qaoa_elev_two.py

To run DIV-QAOA, you can run the following code:

```
python Code/dev-qaoa/div_qaoa_tcm.py [run] [file name]
```
In order to run two case studies related to ELEVATOR, we can run the file loch_qaoa_elev_three.py and loch_qaoa_elev_two.py.

To run GA, you can run the following code:

```
python ga.py [run] [popsize] [file name]
```
Specially, to run two case studies related to ELEVATOR, the "file name" is "elevator_two" and "elevator_three".

Similarly, you can run RS with

```
python rs.py [run] [file name]
```

## Experiment results are saved in [this link](https://drive.google.com/drive/folders/1zvqdwVx5RZeVq1ljI7EoywKN5bWV9BI7?usp=drive_link).
