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
            root = "qaoa_"+str(reps)+"/"+file_name+"/"+"size_"+str(size)+"/"+str(i)+"/"+"solution.csv"
            df = pd.read_csv(root)
            value_list.append(df[feature][0])
        except:
            continue
    return value_list

def get_baseline_value(feature, file_name):# get 10 the value in solution.csv of reps, file_name and size
    if file_name == "elevator_one":
        file_name = "elevator"
    value_list = []
    for i in range(10):
        try:
            root = "divide_baseline_try/"+file_name+"/"+str(i)+"/"+"solution.csv"
            df = pd.read_csv(root)
            value_list.append(df[feature][0])
        except:
            continue
    return value_list


if __name__ == '__main__':
    reps = [1,2,4,6,8,16]
    feature = "best_fval"
    iterations = 10
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    # kruskal test
    for file_name in files:
        feature = "best_fval"
        sizes = [7, 8, 10, 12, 14, 16]
        # reps = [1,2,4,8,16]
        reps = [1,2,4,8,16]
        for j in range(len(sizes)):
            data_list = [[] for _ in range(5)]
            size = sizes[j]
            for i in range(len(reps)):
                data_list[i] += get_solution_value(feature, reps[i], file_name, size)
            # print(data_list)

        H, pval = stats.kruskal(data_list[0], data_list[1], data_list[2], data_list[3], data_list[4])
        print("H-statistic:", H)
        print("P-Value:", pval)
        if pval<0.05:
            print("size: ",size)

    # compare with baseline
    # values of dwave
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
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

    # Convert dictionaries into dataframes
    df1 = pd.DataFrame.from_dict(dic_dwave, orient='index').reset_index().melt(id_vars='index')
    df1.columns = ['Key', 'Group', 'Value']
    df1['Group'] = 'LOCH-QAOA'

    df2 = pd.DataFrame.from_dict(dic_baseline, orient='index').reset_index().melt(id_vars='index')
    df2.columns = ['Key', 'Group', 'Value']
    df2['Group'] = 'DIV-QAOA'

    # Concatenate dataframes
    df = pd.concat([df1, df2])

    # unique keys
    keys = df['Key'].unique()
    #
    # # Create boxplots
    for key in keys:
        plt.figure()
        plt.grid()
        sns.boxplot(x='Group', y='Value', data=df[df['Key'] == key])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.title(key)
        # plt.xlabel('Approach')  # Set x-axis label
        # plt.ylabel('Approximation Ratio')  # Set y-axis label
        plt.show()
        # plt.savefig("analysis/baseline_compare/"+key+"_box.png", dpi=300)
    # print(dic_dwave)

    # heat map
    # file_name = "elevator_two"
    # feature = "best_fval"
    # sizes = [7, 8, 10, 12, 14, 16]
    # # reps = [1,2,4,8,16]
    # reps = [1,2,4,8,16]
    # data_list = [[0 for _ in range(len(sizes))] for _ in range(len(reps))]
    # for size in range(len(sizes)):
    #     for rep in range(len(reps)):
    #         data_list[rep][size] = statistics.mean(get_solution_value(feature, reps[rep], file_name, sizes[size]))
    #
    # ax = sns.heatmap(data_list)
    # ax.set_xlabel("N")
    # ax.set_xticklabels(sizes)
    # ax.set_ylabel("Approximation Ratio")
    # ax.set_yticklabels(reps)
    # plt.savefig("analysis/"+file_name+".png")


    # data_list[rep][size] = statistics.mean(get_solution_value(feature, reps[rep], file_name, sizes[size]))

    # ax = sns.heatmap(data_list)
    # ax.set_xlabel("N")
    # ax.set_xticklabels(sizes)
    # ax.set_ylabel("Approximation Ratio")
    # ax.set_yticklabels(reps)
    # plt.savefig("analysis/"+file_name+".png")

    # # fval
    reps = [1,2,4,6,8,16]
    feature = "best_fval"
    iterations = 10
    files = ["paintcontrol", "iofrol", "gsdtsr", "elevator_two", "elevator_one"]
    # index = 1
    # file_name = files[index]
    # optimals = [0.1408811904392035,0.096699123713009,0.003558040347176,0.0689290441390089, 0.10237206500324063]
    # sizes = [7, 8, 10, 12, 14, 16]
    # for index in range(1,2):
    #     file_name = files[index]
    #     for rep in [reps[0]]:
    #         # Initialize the dictionary with empty lists as values
    #         fval_dic = {size: [] for size in sizes}
    #         for size in sizes:
    #             for i in range(10):
    #                 try:
    #                     root = "qaoa_"+str(rep)+"/"+file_name+"/"+"size_"+str(size)+"/"+str(i)+"/"+"solution.csv"
    #                     df = pd.read_csv(root)
    #                     fval_dic[size].append(df[feature][0]/optimals[index])
    #                 except:
    #                     continue
    #             print(statistics.mean(fval_dic[size]))
    #         # print(fval_dic)
    #         fval_matrix = list(fval_dic.values())
    #         plt.clf()
    #         plt.violinplot(fval_matrix)
    #
    #         # Set x-axis tick labels
    #         plt.xticks(range(1, len(sizes) + 1), sizes, fontsize=16)
    #         plt.yticks(fontsize=15)
    #         # for elevator one
    #         # plt.yticks(np.arange(1.0, 1.0021, 0.0004))
    #         # plt.ylim(0.00350, 0.00400)
    #         # plt.ylim(0.0966, 0.0973)
    #         # plt.ylim(0.140875, 0.14105)
    #
    #         # plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter('%.4f'))
    #         # plt.gca().yaxis.get_major_formatter().set_scientific(False)
    #         plt.show()

            # # Set y-axis label
            # plt.ylabel('Approximation Ratio')
            #
            # # Set plot title
            # plt.title('Boxplot of Sizes')

            # if not os.path.exists(
            #         "analysis" + "/" + file_name +"/"+str(rep)+ "/"+str(iterations) + "/"):
            #     os.makedirs("analysis" + "/" + file_name +"/"+str(rep)+ "/"+str(iterations) + "/")
            # plt.savefig("analysis" + "/" + file_name +"/"+str(rep)+ "/"+str(iterations) + "/" +file_name+ "_best_fval.png", dpi=300)

    # optimal_dic = {"paintcontrol": 0.14088119043920358, "iofrol": 0.09669912371300904}
    # sizes = [7, 8, 10, 12, 14, 16]
    # counts_optimal = []
    # for rep in reps:
    #     # Initialize the dictionary with empty lists as values
    #     for size in sizes:
    #         count = 0
    #         for i in range(10):
    #             try:
    #                 root = "qaoa_"+str(rep)+"/"+file_name+"/"+"size_"+str(size)+"/"+str(i)+"/"+"solution.csv"
    #                 df = pd.read_csv(root)
    #                 print(optimal_dic)
    #                 print(file_name)
    #                 if df[feature][0] == optimal_dic[file_name]:
    #                     count += 1
    #             except:
    #                 continue
    #         counts_optimal.append(count)
    #     print(counts_optimal)
    #     plt.clf()
    #     plt.bar(range(len(counts_optimal)), counts_optimal)

        # Set x-axis tick labels
        # plt.xticks(range(1, len(sizes) + 1), sizes)
        # plt.ylim(0.140875, 0.14105)

        # Set y-axis label
        # plt.ylabel('Counts')
        #
        # # Set plot title
        # # plt.title('Boxplot of Sizes')
        #
        # if not os.path.exists(
        #         "analysis" + "/" + file_name + "/" + str(rep) + "/" + str(iterations) + "/"):
        #     os.makedirs("analysis" + "/" + file_name + "/" + str(rep) + "/" + str(iterations) + "/")
        # plt.savefig("analysis" + "/" + file_name + "/" + str(rep) + "/" + str(iterations) + "/" + "optimal_count.png")

    # # statistical
    # file_name = "elevator_two"
    # feature = "best_fval"
    # sizes = [7, 8, 10, 12, 14, 16]
    # # reps = [1,2,4,8,16]
    # reps = [1,2,4,8,16]
    # data_list = [[0 for _ in range(len(reps))] for _ in range(len(reps))]
    # for i in range(len(reps)):
    #     data1 = []
    #     for i_s in range(len(sizes)):
    #         size1 = sizes[i_s]
    #         data1 += get_solution_value(feature, reps[i], file_name, size1)
    #     for j in range(len(reps)):
    #         data2 = []
    #         for j_s in range(len(sizes)):
    #             size2 = sizes[j_s]
    #             data2 += get_solution_value(feature, reps[j], file_name, size2)
    #         statistic, pvalue, a12_effect_size = stat_test(data1, data2)
    #         data_list[i][j] = pvalue
    # print(data_list)
    # ax = sns.heatmap(data_list)
    # plt.savefig("analysis/corr_"+file_name+".png")




    # elevator two: 10; gsdtsr: 12
    # file_name = "gsdtsr"
    # feature = "best_fval"
    # size = 12
    # # reps = [1,2,4,8,16]
    # reps = [1,2,4,8,16]
    # data_list = [[0 for _ in range(len(reps))] for _ in range(len(reps))]
    # for i in range(len(reps)):
    #     data1 = get_solution_value(feature, reps[i], file_name, size)
    #     for j in range(len(reps)):
    #         data2 = get_solution_value(feature, reps[j], file_name, size)
    #         statistic, pvalue, a12_effect_size = stat_test(data1, data2)
    #         data_list[i][j] = pvalue
    #         if pvalue<0.05:
    #             print(reps[i])
    #             print(data1)
    #             print(reps[j])
    #             print(data2)
    #             print(a12_effect_size)
    #             print("\n")
    # # print(data_list)
    # ax = sns.heatmap(data_list, annot=True)
    # ax.set_xticklabels([1,2,4,8,16])
    # ax.set_yticklabels([1, 2, 4, 8, 16])
    # plt.show()
    # # plt.savefig("analysis/corr_"+file_name+".png")



    #
    # import pandas as pd
    import seaborn as sns

    # # Here are your two dictionaries
    # dict1 = {'key1': [1, 2, 3, 4, 5], 'key2': [1, 2, 3, 4, 5], 'key3': [1, 2, 3, 4, 5]}
    # dict2 = {'key1': [2, 3, 4, 5, 6], 'key2': [2, 3, 4, 5, 6], 'key3': [2, 3, 4, 5, 6]}

    # # Convert dictionaries into dataframes
    # df1 = pd.DataFrame.from_dict(dic_dwave, orient='index').reset_index().melt(id_vars='index')
    # df1.columns = ['Key', 'Group', 'Value']
    # df1['Group'] = 'Dict1'
    #
    # df2 = pd.DataFrame.from_dict(dic_baseline, orient='index').reset_index().melt(id_vars='index')
    # df2.columns = ['Key', 'Group', 'Value']
    # df2['Group'] = 'Dict2'
    #
    # # Concatenate dataframes
    # df = pd.concat([df1, df2])
    #
    # # Create boxplot
    # sns.boxplot(x='Key', y='Value', hue='Group', data=df)
    # plt.show()


    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # Here are your two dictionaries
    # dict1 = {'key1': [1, 2, 3, 4, 5], 'key2': [1, 2, 3, 4, 5], 'key3': [1, 2, 3, 4, 5]}
    # dict2 = {'key1': [2, 3, 4, 5, 6], 'key2': [2, 3, 4, 5, 6], 'key3': [2, 3, 4, 5, 6]}
    #















