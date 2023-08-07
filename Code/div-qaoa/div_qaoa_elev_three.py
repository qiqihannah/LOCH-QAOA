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
import math
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
        #         print("time:",obj_time)
        #         print("rate:",obj_rate)
        #         print("num:",obj_num)

        #         obj_time = sum(self._times[i] * x[i] for i in x)
        #         obj_time = pow(obj_time/time_sum, 2)

        #         if rate_sum == 0:
        #             obj_fr = 0
        #             obj_fr = 0
        #         else:
        #             obj_fr = sum(self._frs[i] * x[i] for i in x)
        #             obj_fr = pow((obj_fr-rate_sum)/rate_sum, 2)

        #         obj_num = pow(sum(x[i] for i in x)/len(self._times), 2)

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

def divide_full(data, sample_size, num):
    '''
    :param data: a list of case numbers. test cases to be divided.
    :param sample_size:
    :return:
    '''

    sample_list_total = []
    initial_flag = True

    cases = data.copy()
    count = 0
    for sub in range(num):
        count += 1
        if sample_size>len(cases):
            sample_list = cases
            diff = sample_size - len(cases)
            data_add = []
            for i in data:
                if i not in sample_list:
                    data_add.append(i)
            sample_list += random.sample(data_add,diff)
            sample_list_total.append(sample_list)
        else:
            sample_list=random.sample(cases,sample_size)
            sample_list_total.append(sample_list)
            for i in sample_list:
                cases.remove(i)
        if count == math.ceil(len(data)/sample_size):
            cases = data.copy()
            count = 0
    return sample_list_total

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
            # print(t[1:]+'. ',end=' ')
            # print('time: '+str(foods[t]['time']), end=', ')
            # print('rate: '+str(foods[t]['rate']), end='\n')
    fval = (1 / 3) * pow(sum(cost_list) / sum(data['cost']), 2) + (1 / 3) * pow((sum(pcount_list) - sum(data["pcount"])) / (sum(data["pcount"])), 2) + (1 / 3) * pow((sum(dist_list) - sum(data['dist'])) / (sum(data["dist"])), 2)
    # print("Total time: " + str(total_time))
    # print("Total rate: " + str(total_rate))
#     print("Fval value:" + str(fval))
#     print("Number: "+str(count))
    return fval

def divide_small(data, sample_size):
    '''
    :param data: a list of case numbers. test cases to be divided.
    :param sample_size:
    :return:
    '''
    sample_list_total = []
    for i in range(math.ceil(len(data)/sample_size)):
        if sample_size>len(data):
            sample_list = data
            sample_list_total.append(sample_list)
        else:
            sample_list=random.sample(data,sample_size)
            sample_list_total.append(sample_list)
            for i in sample_list:
                data.remove(i)
    return sample_list_total

def run_alg(qubo, reps):
    seed = random.randint(1, 9999999)
    algorithm_globals.random_seed = seed
    optimizer = COBYLA(100)
    backend = Aer.get_backend('aer_simulator')
    # backend.set_options(device='GPU')
    quantum_instance = QuantumInstance(backend)
    qaoa_mes = QAOA(quantum_instance=quantum_instance, optimizer=optimizer, include_custom=True, reps=reps)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
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
    time = []
    rate = []
    for t in range(len(solution)):
        if solution[t] == 1.0:
            time.append(data.iloc[t]['time'])
            rate.append(data.iloc[t]['rate'])
    plt.scatter(data["time"], data["rate"], c='red')
    plt.scatter(time, rate)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('name',type=str)
    args = parser.parse_args()
    num_experiment = args.n
    file_name = args.name
    sample_size = 7
    df = pd.DataFrame()
    df = pd.read_csv("data/"+file_name+".csv", dtype={"time": float, "rate": float})
    reps = 1
    total = list(range(0, len(df)))
    dic = {"paintcontrol": 13, "iofrol": 402, "gsdtsr": 976, "elevator_two": 406, "elevator_one": 456}
    num_run = dic["elevator_one"]
    sample_total_list = divide_full(total, sample_size, num_run)
    cost, p_count, dist = get_data(df)
    solution = [random.choice([0, 1]) for _ in range(len(cost))]

    head_log = ["sub_problem","sub_fval", "solution", "qaoa_time"]
    head_solution = ["sub_problem_num", "fval", "solution", "qaoa_total", "exe_total"]
    log_df = pd.DataFrame(columns=head_log)
    solution_df = pd.DataFrame(columns=head_solution)

    log_fval = []
    log_sub = []
    log_solution = []
    log_qaoa_time = []
    count_sub = 0
    qaoa_time_total = 0
    start_exe = time.time()
    for case_list in sample_total_list:
        count_sub += 1
        qubo, testcase = create_qubo(cost, p_count, dist, 1 / 3, 1 / 3, 1 / 3, case_list, solution)
        result, qaoa_time = run_alg(qubo, reps)
        for case_index in range(len(case_list)):
            solution[case_list[case_index]] = result.x[case_index]
        log_df.loc[len(log_df)] = [case_list, result.fval, result.x, qaoa_time]
        qaoa_time_total += qaoa_time
    end_exe = time.time()
    total_time = end_exe - start_exe
    final_fval = print_diet(solution, df)
    solution_df.loc[len(solution_df)] = [count_sub, final_fval, solution, qaoa_time_total, total_time]
    # scatter_merge(solution, df)

    if not os.path.exists("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment)):
        os.makedirs("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment))
    log_df.to_csv("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment)+"/log.csv")

    if not os.path.exists("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment)):
        os.makedirs("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment))
    solution_df.to_csv("divide_baseline_try/"+ file_name + "_three" + "/" + str(num_experiment)+"/solution.csv")