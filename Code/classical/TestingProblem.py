import numpy as np
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.core.solution import BinarySolution
from jmetal.core.problem import BinaryProblem
from jmetal.problem.singleobjective.unconstrained import OneMax
import math
import random


def get_fval(sample,data):
    count = 0
    total_time = 0
    total_rate = 0
    time_list = []
    rate_list = []
    for t in range(len(sample)):
        if sample[t] == 1:
            total_time += data.iloc[t]['time']
            total_rate += data.iloc[t]['rate']
            time_list.append(data.iloc[t]['time'])
            rate_list.append(data.iloc[t]['rate'])
            # print(t[1:]+'. ',end=' ')
            # print('time: '+str(foods[t]['time']), end=', ')
            # print('rate: '+str(foods[t]['rate']), end='\n')
            count += 1
    fval = (1 / 3) * pow(sum(time_list) / sum(data['time']), 2) + (1 / 3) * pow((sum(rate_list) - sum(data["rate"]) + 1e-20) / (sum(data["rate"])+1e-20), 2) + (1 / 3) * pow(count / len(data), 2)
    # print("Total time: " + str(total_time))
    # print("Total rate: " + str(total_rate))
#     print("Fval value:" + str(fval))
#     print("Number: "+str(count))
    return fval

def get_fval_ele_three(sample,data):
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
    fval = (1 / 3) * pow(sum(cost_list) / sum(data['cost']), 2) + (1 / 3) * pow(
        (sum(pcount_list) - sum(data["pcount"])) / (sum(data["pcount"])), 2) + (1 / 3) * pow(
        (sum(dist_list) - sum(data['dist'])) / (sum(data["dist"])), 2)
    # print("Total time: " + str(total_time))
    # print("Total rate: " + str(total_rate))
    #     print("Fval value:" + str(fval))
    #     print("Number: "+str(count))
    return fval

def get_fval_ele_two(sample,data):
    total_cost = 0
    total_div = 0
    cost_list = []
    div_list = []
    for t in range(len(sample)):
        if sample[t] == 1:
            total_cost += data.iloc[t]['cost']
            total_div += data.iloc[t]['input_div']
            cost_list.append(data.iloc[t]['cost'])
            div_list.append(data.iloc[t]['input_div'])
            # print(t[1:]+'. ',end=' ')
            # print('time: '+str(foods[t]['time']), end=', ')
            # print('rate: '+str(foods[t]['rate']), end='\n')
    fval = (1 / 2) * pow(sum(cost_list) / sum(data['cost']), 2) + (1 / 2) * pow((sum(div_list) - sum(data["input_div"])) / (sum(data["input_div"])), 2)
    # print("Total time: " + str(total_time))
    # print("Total rate: " + str(total_rate))
#     print("Fval value:" + str(fval))
#     print("Number: "+str(count))
    return fval

class TCO(BinaryProblem):
    """ The implementation of the OneMax problems defines a single binary variable. This variable
    will contain the bit string representing the solutions.

    """
    def __init__(self, number_of_bits: int = 256, data = [], case_study = ""):
        super(TCO, self).__init__()
        self.number_of_bits = number_of_bits

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["TCO"]
        self.data = data
        self.case_study = case_study

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
        if self.case_study == "elevator_two":
            fval = get_fval_ele_two(selection, self.data)
        elif self.case_study == "elevator_three":
            fval = get_fval_ele_three(selection, self.data)
        else:
            fval = get_fval(selection, self.data)
        solution.objectives[0] = fval
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def name(self) -> str:
        return "TCO"

# class TCO(BinaryProblem):
#
#     def __init__(self, number_of_bits, data_df):
#         super(TCO, self).__init__()
#         self.number_of_bits = number_of_bits
#         self.number_of_objectives = 1
#         self.number_of_variables = 1
#         self.number_of_constraints = 0
#         self.data = data_df
#
#         self.obj_directions = [self.MINIMIZE]
#         self.obj_labels = ['TCO']
#
#     def evaluate(self, solution: BinarySolution) -> BinarySolution:
#         selection = []
#         for bit in solution.variables[0]:
#             if bit:
#                 selection.append(1)
#             else:
#                 selection.append(0)
#         print(selection)
#         fval = print_diet(selection, self.data)
#         print(fval)
#         solution.objectives[0] = fval
#         return solution
#
#     def create_solution(self) -> BinarySolution:
#         new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
#         new_solution.variables[0] = \
#             [True if random.randint(0, 1) == 1 else False for _ in range(self.number_of_bits)]
#         return new_solution
#
#
#     def get_name(self) -> str:
#         return 'TCO'


# class TestingProblem(BinaryProblem):
#     def __init__(self, config_dic, solution_sheet, log_sheet):
#         super(IntegerProblem).__init__()
#
#         n = len(config_dic['inputID'])
#         if 'M' in config_dic.keys():
#             self.number_of_variables = config_dic['M']
#         else:
#             self.number_of_variables = math.ceil(pow(2, n) * config_dic['beta'])
#         self.number_of_objectives = 1
#         self.number_of_constraints = 0
#
#
#         self.obj_directions = [self.MINIMIZE]
#
#         self.lower_bound = self.number_of_variables * [0]
#
#         self.upper_bound = self.number_of_variables * [pow(2,n)-1]
#
#         self.config_dic = config_dic
#         self.solution_sheet = solution_sheet
#         self.log_sheet = log_sheet
#
#
#     def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
#         variables = np.array(solution.variables)
#         p = calculate_fail_number_GA(variables, self.config_dic, self.solution_sheet, self.log_sheet)
#         # the value is negated because we want to maximize "p" using a minimization problem
#         solution.objectives[0] = -p
#         return solution
#
#     def get_name(self) -> str:
#         return 'TestingProblem'
