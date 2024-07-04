# Statistical Results of LOCH-QAOA
### Comparison of LOCH-QAOA with various layers (p)

|             | Paint Control | IOF/ROL | GSDTSR | ELEVATOR_o2 | ELEVATOR_o3 |
|-------------|---------------|---------|--------|-------------|-------------|
| H-statistic | 1.887         | 0.070   | 0.631  | 7.593       | 2.784       |
| P-Value     | 0.757         | 0.999   | 0.960  | 0.108       | 0.595       |

We use Kruskal-Wallis H Test to compare whether results obtained by LOCH-QAOA with different layers have significant difference.

### Comparison of LOCH-QAOA with various sub-problem sizes (N)
We first use Kruskal-Wallis H Test to compare whether there is significant difference between LOCH-QAOA with various sub-problem sizes. Results show that p-values are all smaller than 0.001.

We then use Mann-Whitney U Test as the statistical test and Vargha and Delaney’s A12 statistics as effect size. The p-values and A12 values are shown in the heatmaps under Analyse/graphs/loch-qaoa-compare-sizes. The x and y axises are sub-problem size values.
According to the graphs, LOCH-QAOA with lower sub-problem sizes tend to get lower fitness values (i.e., better performance).

### Comparison of LOCH-QAOA and RS
We then use Mann-Whitney U Test as the statistical test and Vargha and Delaney’s A12 statistics as effect size. The p-values and A12 values are shown in the table below. A p-value less than 0.05 indicates
that there is no significant difference between the ar values
of the two selected groups. In this case, we utilize Vargha
and Delaney’s ˆA12 statistic as the effect size measure to
quantify the magnitude of the difference between the two
groups. If ˆA12 is 0.5, the result is achieved by chance. If ˆA12
is smaller than 0.5, the former corresponding approach per-
forms better than that of the latter one during comparison
and vice versa.

|         | Paint Control | IOF/ROL | GSDTSR | ELEVATOR_o2 | ELEVATOR_o3 |
|---------|---------------|---------|--------|-------------|-------------|
| P-Value | < 1e-3        | < 1e-3   | < 1e-3    | < 1e-3       | < 1e-3       |
| A12     | -             | -       | -      | -           | -           |

### Comparison of LOCH-QAOA and GA
We then use Mann-Whitney U Test as the statistical test and Vargha and Delaney’s A12 statistics as effect size. The p-values and A12 values are shown in the table below.

|         | Paint Control | IOF/ROL | GSDTSR | ELEVATOR_o2 | ELEVATOR_o3 |
|---------|---------------|---------|--------|-------------|-------------|
| P-Value | 1.0           | 0.006   | 1.0    | 0.006       | 0.408       |
| A12     | -             | -       | -      | -           | -           |
