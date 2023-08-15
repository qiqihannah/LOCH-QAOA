import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import statistics
from scipy.stats import mannwhitneyu
from scipy import stats
import seaborn as sns

def get_value(num, feature, file, size):
    root = "LOCH-QAOA-result/qaoa_1/" + file + "/" + "size_" + str(size)+ "/"+ str(num) + "/itr_results.csv"
    df = pd.read_csv(root)
    return list(df[feature])

def get_log_value(num, feature, file, size):
    root = "LOCH-QAOA-result/qaoa_1/" + file + "/" + "size_" + str(size)+ "/"+ str(num) + "/log.csv"
    df = pd.read_csv(root)
    return list(df[feature])

def get_baseline_value(num, feature, file, size):
    root = "Div-QAOA-result/" + file + "/" + "/"+ str(num) + "/log.csv"
    df = pd.read_csv(root)
    return list(df[feature])

def find_first_non_decreasing_point(lst):
    # Check the input
    if len(lst) < 3:
        return None

    # Go through the list
    for i in range(2, len(lst)):
        if lst[i-2] <= lst[i-1] and lst[i-2] <= lst[i]:
            return i  # Return the index immediately when the condition is met
    return None  # Return None if such index is not found

def get_optimal(feature, file, size):
    root = "LOCH-QAOA-result/qaoa_1/" + file + "/" + "size_" + str(size) + "/"+ "/optimal.csv"
    df = pd.read_csv(root)
    return list(df[feature])

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



if __name__ == '__main__':
    # gengerating optimal files
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_three"]
    optimals = [0.1408811904392035,0.096699123713009,0.003558040347176,0.0689290441390089, 0.10237206500324063]
    sizes = [7,8,10,12,14,16]
    for file in files:
        for size in sizes:
            stopping_index_list = []
            qaoa_time_list = []
            total_time_list = []
            optimal_fval_list = []
            eve_value_list = []
            for num in range(10):
                fval_list = get_value(num, "fval", file, size)
                stopping_index = find_first_non_decreasing_point(fval_list)
                optimal_fval_list.append(fval_list[stopping_index-2])
                qaoa_list = get_value(num, "qaoa_total", file, size)
                exe_list = get_value(num, "exe_total", file, size)
                eve_list = get_value(num, "exe_count", file, size)
                eve_value_list.append(sum(eve_list[:stopping_index+1]))
                qaoa_time_list.append(sum(qaoa_list[:stopping_index+1]))
                total_time_list.append(sum(exe_list[:stopping_index+1]))
                percentage_list = [qaoa_time_list[k]/total_time_list[k] for k in range(len(qaoa_time_list))]
                stopping_index_list.append(stopping_index+1)
            data_dic = {}
            data_dic["fval"] = optimal_fval_list
            data_dic["eve"] = eve_value_list
            data_dic["optimal_itr"] = stopping_index_list
            data_dic["total_time"] = total_time_list
            data_dic["qaoa_time"] = qaoa_time_list
            data_dic["percentage"] = percentage_list

            df = pd.DataFrame(data_dic)
            df.to_csv("LOCH-QAOA-result/qaoa_1/" + file + "/size_"+str(size)+"/"+"optimal.csv")

    # executing time
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    sizes = [7, 8, 10, 12, 14, 16]
    for file in files:
        for size in sizes:
            qaoa_time_list = get_optimal("total_time", file, size)
            percentage = get_optimal("percentage", file, size)
            qaoa_time_mean = statistics.mean(qaoa_time_list)
            qaoa_time_std = statistics.stdev(qaoa_time_list)
            percentage_mean = statistics.mean(percentage)
            percentage_mean = percentage_mean*100
            print("{:.1f}".format(qaoa_time_mean), end="")
            print("±",end="")
            print("{:.1f}".format(qaoa_time_std), end=" ")
            print("{:.1f}%".format(percentage_mean), end=" ")
        print("\n")


    # numEva
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    size = 7
    for file in files:
        eve_list = get_optimal("eve", file, size)
        print("{:.1f}".format(statistics.mean(eve_list)),end="")
        print("±", end="")
        print("{:.1f}".format(statistics.stdev(eve_list)),end="")
        print("\n")


    # draw div-qaoa and loch-qaoa trend by number of evaluations
    for file_index in range(len(files)):
        file = files[file_index]
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        index_list = []
        fval_list = []
        size = 7
        for num in range(10):
            fvals_baseline = [i/optimals[file_index] for i in get_baseline_value(num, "sub_fval", file, size)]
            fvals = [j/optimals[file_index] for j in get_log_value(num, "fval", file, size)]
            fvals = fvals[:len(fvals_baseline)]
            plt.plot(list(range(1, len(fvals) + 1)), fvals_baseline, color=cm.viridis(1. * 1 / 6),alpha=0.5)
            plt.plot(list(range(1,len(fvals)+1)), fvals, color=cm.viridis(1.*3/6),alpha=0.5)
        custom_lines = [mlines.Line2D([0], [0], color=cm.viridis(1. * 3 / 6), lw=4), mlines.Line2D([0], [0], color=cm.viridis(1. * 1 / 6), lw=4)]
        labels = ["LOCH-QAOA","Div-QAOA"]
        plt.legend(custom_lines, labels, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        # plt.show()
        plt.savefig("analysis/baeline_temp/"+file+"_trend"+".png", dpi=300)
        plt.clf()

    # draw loch-qaoa trend by iterations
    for k in range(5):
        file = files[k]
        optimal = optimals[k]
        labels = ['N=7', 'N=8', 'N=10', 'N=12', 'N=14', 'N=16']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        max_x = 0
        for size_id in range(len(sizes)):
            for num in range(10):
                fvals = get_value(num, "fval", file, sizes[size_id])
                fvals = [i/optimal for i in fvals]
                stopping_index = find_first_non_decreasing_point(fvals)
                index_list=list(range(1,stopping_index+2))
                fval_list=fvals[:stopping_index+1]
                plt.plot(index_list, fval_list, color=cm.viridis(1.*size_id/6), alpha=0.3, label=labels[size_id])
                if max(index_list)>max_x:
                    max_x = max(index_list)
            custom_lines = [mlines.Line2D([0], [0], color=cm.viridis(1.*i/6), lw=4) for i in range(6)]
            plt.grid(True)
            plt.legend(custom_lines, labels, fontsize=14)
            plt.xticks(range(1, max_x + 1), fontsize=15)
            plt.yticks(fontsize=16)

        plt.savefig("LOCH-QAOA-result/qaoa_1/graph/"+file+"_trend.png", dpi=300,bbox_inches='tight')
        plt.clf()

    # kruskal test for different sub-problem sizes
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    feature = "best_fval"
    sizes = [7, 8, 10, 12, 14, 16]
    for file in files:
        data_list = [[] for _ in range(len(sizes))]
        for j in range(len(sizes)):
            size = sizes[j]
            data_list[j] = get_optimal("fval", file, size)
        try:
            H, pval = stats.kruskal(data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5])
            print(file)
            print("H-statistic:", H)
            print("P-Value:", pval)
            if pval<0.05:
                print("size: ",size)
        except:
            print("wrong")

    for file_id in range(len(files)):
        feature = "best_fval"
        pvalue_list = [[0 for _ in range(len(sizes))] for _ in range(len(sizes))]
        a12_list = [[0 for _ in range(len(sizes))] for _ in range(len(sizes))]
        for size_id in range(len(sizes)):
            size = sizes[size_id]
            data_list[size_id] = get_optimal("fval", file, size)
        for i in range(len(sizes)):
            data1 = data_list[i]
            for j in range(len(sizes)):
                data2 = data_list[j]
                statistic, pvalue, a12_effect_size = stat_test(data1, data2)
                pvalue_list[i][j] = pvalue
                a12_list[i][j] = a12_effect_size
                if pvalue<0.05:
                    print(sizes[i])
                    print(sizes[j])
                    print(a12_effect_size)
                    print("\n")
        # print(data_list)
        ax = sns.heatmap(pvalue_list, annot=True)
        ax.set_xticklabels(sizes)
        ax.set_yticklabels(sizes)
        plt.show()
        plt.clf()

        ax = sns.heatmap(a12_list, annot=True)
        ax.set_xticklabels(sizes)
        ax.set_yticklabels(sizes)
        plt.show()
        plt.clf()
