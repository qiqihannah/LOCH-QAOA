import csv
import os.path
import time
from jmetal.util.observer import Observer
from jmetal.core.solution import Solution
import logging
LOGGER = logging.getLogger("jmetal")
from matplotlib import pyplot as plt

class CuBasicObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""

        self.display_frequency = frequency
        self.evaluation_list = []
        self.fitness_list = []
        self.computing_time_list = []
        self.solution_list = []

    def update(self, *args, **kwargs):
        computing_time = kwargs["COMPUTING_TIME"]
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
            )
            self.evaluation_list.append(evaluations)
            self.fitness_list.append(fitness[0])
            self.computing_time_list.append(computing_time)
        solution_bit = []
        for value in solutions.variables[0]:
            if value:
                solution_bit.append(1)
            else:
                solution_bit.append(0)
        self.solution_list.append(solution_bit)
    def export_to_csv(self, file_name):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time", "Solution"])
            writer.writerows(
                zip(self.evaluation_list, self.fitness_list, self.computing_time_list, self.solution_list)
            )
    def export_gen(self, file_name):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time"])
            writer.writerows(
                zip(self.evaluation_list, self.fitness_list, self.computing_time_list)
            )
    def export_plot(self, file_name):
        plt.clf()
        evaluation_list_gen = [self.evaluation_list[i] for i in range(len(self.evaluation_list))]
        plt.plot(evaluation_list_gen, self.fitness_list)
        plt.savefig(file_name)

    def export_form(self, file_name, num, pop, mu, crossover):
        file_exists = os.path.isfile(file_name)
        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Index", "Population", "Mutation Rate", "Crossover Rate", "Fitness Value", "Computing Time"])
                writer.writerow([num, pop, mu, crossover, self.fitness_list[-1], self.computing_time_list[-1]])
            writer.writerow([num, pop, mu, crossover, self.fitness_list[-1], self.computing_time_list[-1]])
