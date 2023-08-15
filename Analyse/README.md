# Statistical Results of LOCH-QAOA
### Comparison of LOCH-QAOA with various layers (p)

|             | Paint Control | IOF/ROL | GSDTSR | ELEVATOR_o2 | ELEVATOR_o3 |
|-------------|---------------|---------|--------|-------------|-------------|
| H-statistic | 6.447         | 1.752   | 2.917  | 2.184       | 2.836       |
| P-Value     | 0.168         | 0.781   | 0.572  | 0.702       | 0.586       |
We use Kruskal-Wallis H Test to compare whether results obtained by LOCH-QAOA with different layers have significant difference.

### Comparison of LOCH-QAOA with various sub-problem sizes (N)
We first use Kruskal-Wallis H Test to compare whether there is significant difference between LOCH-QAOA with various sub-problem sizes. Results show that p-values are all smaller than 0.001.

We then use Mann-Whitney U Test as the statistical test and Vargha and Delaney’s A12 statistics as effect size. The p-values and A12 values are shown in the heatmaps under Analyse/graphs/loch-qaoa-compare-sizes. The x and y axises are sub-problem size values.
According to the graphs, LOCH-QAOA with lower sub-problem sizes tend to get lower fitness values (i.e., better performance).


