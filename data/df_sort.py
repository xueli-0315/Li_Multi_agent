import pandas as pd
df = pd.read_parquet('data/panel_data.parquet')

# 将数据按 时间 和 品种 进行逻辑排序
df_sorted = df.sort_index()

# 查看排序后的效果
print(df_sorted.head(10))

# 检查时间轴是否有断档
print("\n时间范围：", df.index.get_level_values('datetime').min(), "至", df.index.get_level_values('datetime').max())
