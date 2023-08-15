from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.result import QuasiDistribution
from typing import List, Union
from qiskit.quantum_info import Pauli, Statevector
from qiskit.result import QuasiDistribution
import numpy as np
from docplex.mp.model import Model
from qiskit.algorithms import QAOA

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
import pandas as pd
import random
import sys

# from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, GradientDescent
from qiskit.primitives import Sampler
import argparse
import time
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestCaseOptimizationThree(OptimizationApplication):
    """Optimization application for the "knapsack problem" [1].

    References:
        [1]: "Knapsack problem",
        https://en.wikipedia.org/wiki/Knapsack_problem
    """

    def __init__(self, cost: List[float], pcount: List[float], dist: List[float], w1: float, w2: float, w3: float, sample: List[int],
                 solution: List[int]) -> None:
        """
        Args:
            values: A list of the values of items
            weights: A list of the weights of items
            max_weight: The maximum weight capacity
        """
        self._cost = cost
        self._pcount = pcount
        self._dist = dist
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3
        self._sample = sample
        self._solution = solution

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a knapsack problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the knapsack problem instance.
        """
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in self._sample}

        obj_cost = 0
        obj_pcount = 0
        obj_dist = 0

        #         dic_clamp = {}
        for i in range(len(self._solution)):
            if i in self._sample:
                obj_cost += self._cost[i] * x[i]
                obj_pcount += self._pcount[i] * x[i]
                obj_dist += self._dist[i] * x[i]
            else:
                obj_cost += self._cost[i] * self._solution[i]
                obj_pcount += self._pcount[i] * self._solution[i]
                obj_dist += self._dist[i] * self._solution[i]

        cost_sum = sum(self._cost)
        pcount_sum = sum(self._pcount)
        dist_sum = sum(self._dist)

        obj_cost = pow(obj_cost / cost_sum, 2)
        obj_pcount = pow((obj_pcount - pcount_sum) / pcount_sum, 2)
        obj_dist = pow((obj_dist - dist_sum) / dist_sum, 2)

        obj = self._w1 * obj_cost + self._w2 * obj_pcount + self._w3 * obj_dist

        mdl.minimize(obj)

        op = from_docplex_mp(mdl)

        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as item indices

        Args:
            result : The calculated result of the problem

        Returns:
            A list of items whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, value in enumerate(x) if value]

def create_qubo(cost, pcount, dist, w1, w2, w3, sample, solution):
    testcase = TestCaseOptimizationThree(cost, pcount, dist, w1, w2, w3, sample, solution)
    prob = testcase.to_quadratic_program()
    probQubo = QuadraticProgramToQubo() #parameter: cofficient for constraint
    qubo = probQubo.convert(prob)
    return qubo, testcase

def get_data(data):
    cost = data["cost"].values.tolist()
    p_count = data["pcount"].values.tolist()
    dist = data["dist"].values.tolist()
    return cost, p_count, dist

def print_diet(sample,data):
    total_cost = 0
    total_pcount = 0
    total_dist = 0
    cost_list = []
    pcount_list = []
    dist_list = []
    for t in range(len(sample)):
        if sample[t] == 1:
            total_cost += data.iloc[t]['cost']
            total_pcount += data.iloc[t]['pcount']
            total_dist += data.iloc[t]['dist']
            cost_list.append(data.iloc[t]['cost'])
            pcount_list.append(data.iloc[t]['pcount'])
            dist_list.append(data.iloc[t]['dist'])
    fval = (1 / 3) * pow(sum(cost_list) / sum(data['cost']), 2) + (1 / 3) * pow((sum(pcount_list) - sum(data["pcount"])) / (sum(data["pcount"])), 2) + (1 / 3) * pow((sum(dist_list) - sum(data['dist'])) / (sum(data["dist"])), 2)
    return fval

def OrderByImpact(best_solution, df, best_energy):
    impact_values = {}
    for case in range(len(best_solution)):
        if best_solution[case] == 1:
            temp = best_solution.copy()
            temp[case] = 0
            impact_values[case] = 1
            impact_values[case] = print_diet(temp, df) - best_energy
        elif best_solution[case] == 0:
            temp = best_solution.copy()
            temp[case]=1
            impact_values[case] = print_diet(temp, df) - best_energy
    print(impact_values)
    impact_values = sorted(impact_values.items(), key = lambda kv:(kv[1], kv[0]))
    impact_list = []
    for case in impact_values:
        impact_list.append(case[0])
    return impact_list

def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    cost_array = list(df["cost"])
    pcount_array = list(df["pcount"])
    dist_array = list(df["dist"])
    cost_matrix = np.array(cost_array).reshape(-1, 1)
    pcount_matrix = np.array(pcount_array).reshape(-1, 1)
    dist_matrix = np.array(dist_array).reshape(-1, 1)
    matrix = np.array([best_solution] * len(best_solution))
    for i in range(num):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        elif matrix[i][i] == 1:
            matrix[i][i] = 0
    cost_sum = sum(cost_array)
    pcount_sum = sum(pcount_array)
    dist_sum = sum(dist_array)
    cost_obj = matrix.dot(cost_matrix)
    pcount_obj = matrix.dot(pcount_matrix) - pcount_sum
    dist_obj = matrix.dot(dist_matrix) - dist_sum
    obj = (1/3)*(cost_obj/cost_sum)**2 + (1/3)*(pcount_obj/pcount_sum)**2 + (1/3)*(dist_obj/dist_sum)**2 - best_energy
    print(obj)
    # Get the sorted indices
    sorted_indices = np.argsort(obj, axis=0)

    # Convert the sorted indices to a flattened array
    sorted_indices = sorted_indices.flatten()

    return sorted_indices


def run_alg(qubo, reps):
    seed = random.randint(1, 9999999)
    algorithm_globals.random_seed = seed
    optimizer = COBYLA(100)
    backend = Aer.get_backend('aer_simulator')
    # backend.set_options(device='GPU')
    quantum_instance = QuantumInstance(backend)
    qaoa_mes = QAOA(quantum_instance=quantum_instance, optimizer=optimizer, include_custom=True, reps=reps)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    op, offset = qubo.to_ising()
    # Ising formulation
    print("offset: {}".format(offset))
    print("Ising Hamiltonian:")
    print(op)
    begin = time.time()
    qaoa_result = qaoa.solve(qubo)
    end = time.time()
    exe_time = end-begin
    return qaoa_result, exe_time

def print_result(result, testcase):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = testcase.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

def plot(fval_list, reps, file_name, problem_size):
    plt.plot(fval_list)
    plt.ylabel('fval')
    plt.savefig("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)+"/fval_trend.png")

def scatter_merge(solution, data):
    # time = []
    # rate = []
    # for t in range(len(solution)):
    #     if solution[t] == 1.0:
    #         time.append(data.iloc[t]['time'])
    #         rate.append(data.iloc[t]['rate'])
    # plt.scatter(data["time"], data["rate"], c='red')
    # plt.scatter(time, rate)
    # plt.show()
    test_all = []
    test_sel = []
    for t in range(len(solution)):
        test_all.append([data.iloc[t]['cost'], data.iloc[t]['pcount'], data.iloc[t]['dist']])
        if solution[t] == 1.0:
            test_sel.append([data.iloc[t]['cost'], data.iloc[t]['pcount'], data.iloc[t]['dist']])
    print(test_sel)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot unselected data
    ax.scatter(*zip(*test_all), color='r', label='Unselected')

    # Plot selected data
    ax.scatter(*zip(*test_sel), color='g', label='Selected')



    # Label axes
    ax.set_xlabel('cost')
    ax.set_ylabel('passenger count')
    ax.set_zlabel('travel distance')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def get_initial_fval(length):
    initial_values = [random.choice([0, 1]) for _ in range(length)]
    fval = print_diet(initial_values, df)
    best_solution=initial_values
    best_energy=fval
    return best_solution, best_energy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('s', type=int)
    parser.add_argument('p', type=int)
    parser.add_argument('name',type=str)
    args = parser.parse_args()
    num_experiment = args.n
    reps = args.p
    problem_size = args.s
    file_name = args.name
    df = pd.DataFrame()
    df = pd.read_csv("../../data/"+file_name+".csv", dtype={"time": float, "rate": float})
    length = len(df)
    best_solution, best_energy = get_initial_fval(length)
    best_itr = 0
    start_impact = time.time()
    impact_order = OrderByImpactNum(best_solution, df, best_energy)
    end_impact = time.time()
    impact_time = end_impact - start_impact
    index_end = problem_size
    index_begin = 0
    solution = best_solution.copy()
    count = 0 #iteration count
    fval_list = []
    cost, pcount, dist = get_data(df)
    head_log = ["itr_num","sub_problem","fval", "solution", "best_fval", "best_solution", "qaoa_time"]
    head_result = ["itr_num", "exe_count", "fval", "solution", "best_fval", "best_solution", "qaoa_total", "impact_time","exe_total"]
    head_solution = ["best_itr", "best_fval", "best_solution", "total_qaoa", "total_impact","total_exe"]
    log_df = pd.DataFrame(columns=head_log)
    result_df = pd.DataFrame(columns=head_result)
    solution_df = pd.DataFrame(columns=head_solution)
    total_qaoa = 0
    total_exe = 0
    total_impact = 0

    itr_num = 0 # number of iterations

    while count < 30:
        df_time = 0 # time for writing experiment results in dataframe, to delete in total running time
        qaoa_time_total = 0 #total running time
        exe_count = 0 #number of sub-problems in one iteration
        itr_num += 1
        total_start = time.time() #total running time start
        if problem_size>0.15*len(df):
            exe_count += 1
            case_list = impact_order[index_begin:index_end]
            qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
            result, qaoa_time = run_alg(qubo, reps)
            start_df = time.time() #dataframe loading time start
            qaoa_time_total += qaoa_time
            origin_solution = []
            for case in case_list:
                origin_solution.append(solution[case])
            for case_index in range(len(case_list)):
                solution[case_list[case_index]] = result.x[case_index]
            fval_list.append(result.fval) # fitness values of all subproblems
            values_log = [itr_num, case_list, result.fval, solution, best_energy, best_solution, qaoa_time]
            log_df.loc[len(log_df)] = values_log #getting log information of one sub-problem
            end_df = time.time()
            df_time += end_df - start_df
        else:
            while index_end <= 0.15 * len(df):
                exe_count += 1
                case_list = impact_order[index_begin:index_end]
                qubo, testcase = create_qubo(cost, pcount, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
                result, qaoa_time = run_alg(qubo, reps)
                start_df = time.time()
                qaoa_time_total += qaoa_time # time of running qaoa
                origin_solution = []
                for case in case_list:
                    origin_solution.append(solution[case])
                for case_index in range(len(case_list)):
                    solution[case_list[case_index]] = result.x[case_index]
                index_begin += problem_size
                index_end += problem_size
                values_log = [itr_num, case_list, result.fval, solution, best_energy, best_solution, qaoa_time]
                log_df.loc[len(log_df)] = values_log # get log information of one sub-problem
                # print("case:" + str(case_list))
                # print("origin_solution:" + str(origin_solution))
                # print("fval:" + str(result.fval))
                # print("value:" + str(result.x))
                fval_list.append(result.fval) # fitness values of all subproblems
                end_df = time.time()
                df_time += end_df - start_df
        energy = result.fval # overall fitness value after running the last sub-problem in one iteration
        if energy < best_energy:
            best_itr = itr_num
            best_solution = solution
            best_energy = energy
        total_end = time.time()
        total_itr_time = total_end - total_start - df_time + impact_time # total execution time in one iteration

        #total time in solution file
        total_qaoa += qaoa_time_total
        total_exe += total_itr_time # total execution in all loops
        total_impact += impact_time

        values_result = [itr_num, exe_count, energy, solution, best_energy, best_solution, qaoa_time_total, impact_time, total_itr_time]
        result_df.loc[len(result_df)] = values_result # results of one iteration

        start_impact= time.time()
        impact_order = OrderByImpactNum(solution, df, energy)
        end_impact = time.time()
        impact_time = end_impact - start_impact
        print("best:" + str(best_energy))
        count += 1
        index_begin = 0
        index_end = problem_size

    values_solution = [best_itr, best_energy, best_solution, total_qaoa, total_impact, total_exe]
    solution_df.loc[len(solution_df)] = values_solution

    if not os.path.exists("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)):
        os.makedirs("qaoa_"+str(reps)+"/" + file_name + "_three"+ "/size_" + str(problem_size) + "/" + str(num_experiment))
    log_df.to_csv("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)+"/log.csv")

    if not os.path.exists("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)):
        os.makedirs("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment))
    result_df.to_csv("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)+"/itr_results.csv")

    if not os.path.exists("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)):
        os.makedirs("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment))
    solution_df.to_csv("qaoa_"+str(reps)+"/" + file_name + "_three" + "/size_" + str(problem_size) + "/" + str(num_experiment)+"/solution.csv")

    plot(fval_list, reps, file_name, problem_size)

