import pandas as pd

# Read results
task_1_df = pd.read_excel("results_task_1.xlsx")
task_2_df = pd.read_excel("results_task_2.xlsx")

common_columns = task_1_df.columns.intersection(task_2_df.columns)
# Drop the common columns from df2
task_2_df = task_2_df.drop(columns=common_columns)

# Same label mapping in both (alphabetically ordered using LabelEncoder)
mapping_df = pd.read_excel("results_task_1.xlsx", sheet_name=1)

# Dataframes are on the same order, just concat the values columns to the task_1_df
df = pd.concat([task_1_df, task_2_df], axis=1)

# Write to excel
with pd.ExcelWriter(f"results.xlsx", engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Results", index=False)
    mapping_df.to_excel(writer, sheet_name="Label Mapping", index=False)