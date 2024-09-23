import pandas as pd
from scipy.stats import pearsonr

from src.database import METRICS

all_metrics = METRICS.find({})

df = pd.DataFrame.from_records(list(all_metrics), index="identifier")

# print(df["success_rate"].to_numpy())
# print(df["mean_alignment_all"].to_numpy())

success = df["success_rate"].to_numpy()
alignment_all = df["mean_alignment_all"].to_numpy()
alignment_success = df["mean_alignment_success"].to_numpy()
mean_plan_size = df["mean_plan_size"].to_numpy()

print(pearsonr(success, alignment_all))
print(pearsonr(success, alignment_success))
print(pearsonr(alignment_all, alignment_success))
print(pearsonr(success, mean_plan_size))
print(pearsonr(mean_plan_size, alignment_all))

# PearsonRResult(statistic=-0.370642286387261, pvalue=0.01855471914210271)
# PearsonRResult(statistic=0.05979230647264386, pvalue=0.7139946179963783)
# PearsonRResult(statistic=0.8566387846489929, pvalue=1.780570894944358e-12)
# PearsonRResult(statistic=0.6159414377871959, pvalue=2.3280423395604994e-05)
# PearsonRResult(statistic=-0.7070814565400081, pvalue=3.392050663625305e-07)
