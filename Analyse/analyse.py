import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import statistics
from scipy.stats import mannwhitneyu
from scipy import stats
def a12(lst1,lst2,rev=True):
  "how often is x in lst1 more than y in lst2?"
  more = same = 0.0
  for x in lst1:
    for y in lst2:
      if x==y : same += 1
      elif rev and x > y : more += 1
      elif not rev and x < y : more += 1
  return (more + 0.5*same)  / (len(lst1)*len(lst2))

def stat_test(app1, app2):
    statistic, pvalue = mannwhitneyu(app1, app2, alternative='two-sided')

    # Calculate the A12 effect size using Vargha and Delaney's formula
    a12_effect_size = a12(app1, app2)
    return statistic, pvalue, a12_effect_size


def get_solution_value(feature, reps, file_name, size):# get 10 the value in solution.csv of reps, file_name and size
    value_list = []
    for i in range(10):
        try:
            root = "LOCH-QAOA-result/qaoa_"+str(reps)+"/"+file_name+"/"+"size_"+str(size)+"/"+str(i)+"/"+"solution.csv"
            df = pd.read_csv(root)
            value_list.append(df[feature][0])
        except:
            continue
    return value_list

def get_baseline_value(feature, file_name):# get 10 the value in solution.csv of reps, file_name and size
    value_list = []
    for i in range(10):
        try:
            root = "Div-QAOA-result/"+file_name+"/"+str(i)+"/"+"solution.csv"
            df = pd.read_csv(root)
            value_list.append(df[feature][0])
        except:
            continue
    return value_list


if __name__ == '__main__':
    # Statistical tests for LOCH-QAOA with different layers
    reps = [1,2,4,6,8,16]
    feature = "best_fval"
    iterations = 10
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    # kruskal test
    for file_name in files:
        feature = "best_fval"
        sizes = [7, 8, 10, 12, 14, 16]
        reps = [1,2,4,8,16]
        for j in range(len(sizes)):
            data_list = [[] for _ in range(5)]
            size = sizes[j]
            for i in range(len(reps)):
                data_list[i] += get_solution_value(feature, reps[i], file_name, size)

        H, pval = stats.kruskal(data_list[0], data_list[1], data_list[2], data_list[3], data_list[4])
        print("H-statistic: {:.3f}".format(H))
        print("P-Value: {:.3f}".format(pval))
        if pval<0.05:
            print("size: ",size)

    # statistical tests to compare LOCH-QAOA with baseline DIV-QAOA
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_three"]
    optimals = [0.1408811904392035, 0.096699123713009, 0.003558040347176, 0.0689290441390089, 0.10237206500324063]
    size = 7
    dic_dwave = {}
    for i in range(len(files)):
        file = files[i]
        fval_list = get_solution_value("best_fval", 1, file, 7)
        fval_list = [fval_list[_]/optimals[i] for _ in range(len(fval_list))]
        dic_dwave[file] = fval_list
    print(dic_dwave)
    #
    dic_baseline = {}
    for i in range(len(files)):
        file = files[i]
        fval_list = get_baseline_value("fval", file)
        fval_list = [fval_list[_] / optimals[i] for _ in range(len(fval_list))]
        dic_baseline[file] = fval_list
    print(dic_baseline)
    for file in files:
        plt.violinplot([dic_dwave[file], dic_baseline[file]])
        plt.xticks([1,2],["LOCH-QAOA","DIV-QAOA"],fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig("analysis/baseline_compare/" + file + "_violin.png", dpi=300, bbox_inches='tight')
        plt.clf()