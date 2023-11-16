import numpy as np
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.core.solution import BinarySolution
from jmetal.core.problem import BinaryProblem
from jmetal.problem.singleobjective.unconstrained import OneMax
import math
import random

def fval_calcuate(sample, data_array, file_name):
    num = len(sample)
    fea_n = len(data_array)
    if fea_n == 2:
        cost_matrix = np.array(data_array[0]).reshape(-1, 1)
        reward_matrix = np.array(data_array[1]).reshape(-1, 1)
        matrix = np.array(sample)
        cost_sum = sum(data_array[0])
        reward_sum = sum(data_array[1])
        cost_obj = matrix.dot(cost_matrix)
        reward_obj = matrix.dot(reward_matrix) - reward_sum
        if file_name == "elevator_two":
            obj = (1 / 2) * (cost_obj / cost_sum) ** 2 + (1 / 2) * (reward_obj / (reward_sum)) ** 2
            return obj[0]
        else:
            reward_obj += 1e-20
            num_matrix = np.full((num, 1), 1)
            num_obj = matrix.dot(num_matrix)
            obj = (1 / 3) * (cost_obj / cost_sum) ** 2 + (1 / 3) * (reward_obj / (reward_sum + 1e-20)) ** 2 + (1 / 3) * (
                        (num_obj) / num) ** 2
            return obj[0]
    else:
        cost_matrix = np.array(data_array[0]).reshape(-1, 1)
        pcount_matrix = np.array(data_array[1]).reshape(-1, 1)
        dist_matrix = np.array(data_array[2]).reshape(-1, 1)
        cost_sum = sum(data_array[0])
        pcount_sum = sum(data_array[1])
        dist_sum = sum(data_array[2])
        matrix = np.array(sample)
        cost_obj = matrix.dot(cost_matrix)
        pcount_obj = matrix.dot(pcount_matrix) - pcount_sum
        dist_obj = matrix.dot(dist_matrix) - dist_sum
        obj = (1 / 3) * (cost_obj / cost_sum) ** 2 + (1 / 3) * (pcount_obj / pcount_sum) ** 2 + (1 / 3) * (
                    dist_obj / dist_sum) ** 2
        return obj[0]


def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    time_array = list(df["time"])
    rate_array = list(df["rate"])
    time_matrix = np.array(time_array).reshape(-1, 1)
    rate_matrix = np.array(rate_array).reshape(-1, 1)
    num_matrix = np.full((num,1), 1)
    matrix = np.array([best_solution] * len(best_solution))
    for i in range(num):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        elif matrix[i][i] == 1:
            matrix[i][i] = 0
    time_sum = sum(time_array)
    rate_sum = sum(rate_array)
    # time_sum_con = np.full((len(best_solution), 1), time_sum)
    # rate_sum_con = np.full((len(best_solution), 1), rate_sum)
    time_obj = matrix.dot(time_matrix)
    rate_obj = matrix.dot(rate_matrix) - rate_sum + 1e-20
    num_obj = matrix.dot(num_matrix)
    obj = (1/3)*(time_obj/time_sum)**2 + (1/3)*(rate_obj/(rate_sum+1e-20))**2 + (1/3)*((num_obj)/len(best_solution))**2 - best_energy
    # Get the sorted indices
    sorted_indices = np.argsort(obj, axis=0)

    # Convert the sorted indices to a flattened array
    sorted_indices = sorted_indices.flatten()

    return sorted_indices


def OrderByImpactNum(best_solution, df, best_energy):
    num = len(best_solution)
    cost_array = list(df["cost"])
    div_array = list(df["input_div"])
    cost_matrix = np.array(cost_array).reshape(-1, 1)
    div_matrix = np.array(div_array).reshape(-1, 1)
    matrix = np.array([best_solution] * len(best_solution))
    for i in range(num):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        elif matrix[i][i] == 1:
            matrix[i][i] = 0
    cost_sum = sum(cost_array)
    div_sum = sum(div_array)
    # time_sum_con = np.full((len(best_solution), 1), time_sum)
    # rate_sum_con = np.full((len(best_solution), 1), rate_sum)
    cost_obj = matrix.dot(cost_matrix)
    div_obj = matrix.dot(div_matrix) - div_sum
    obj = (1/2)*(cost_obj/cost_sum)**2 + (1/2)*(div_obj/div_sum)**2 - best_energy
    # Get the sorted indices
    sorted_indices = np.argsort(obj, axis=0)

    # Convert the sorted indices to a flattened array
    sorted_indices = sorted_indices.flatten()

    return sorted_indices

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
    # time_sum_con = np.full((len(best_solution), 1), time_sum)
    # rate_sum_con = np.full((len(best_solution), 1), rate_sum)
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


class TCO(BinaryProblem):
    """ The implementation of the OneMax problems defines a single binary variable. This variable
    will contain the bit string representing the solutions.

    """
    def __init__(self, number_of_bits: int = 256, data_array = [], file_name = ""):
        super(TCO, self).__init__()
        self.number_of_bits = number_of_bits

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Ones"]
        self.data_array = data_array
        self.file_name = file_name

    def number_of_variables(self) -> int:
        return 1

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        selection = []
        for bit in solution.variables[0]:
            if bit:
                selection.append(1)
            else:
                selection.append(0)
        fval = fval_calcuate(selection, self.data_array, self.file_name)
        solution.objectives[0] = fval
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def name(self) -> str:
        return "OneMax"