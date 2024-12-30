import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df, metric1, metric2):
  """Visualize the distribution of the specified metric across another metric"""
  plt.figure(figsize=(12, 6))
  sns.boxplot(x=metric2, y=metric1, data=df)
  plt.title(f'{metric1} Distribution {metric2}')
  plt.xticks(rotation=45)
  plt.show()

def perform_anova(df, metric1, metric2):
  """Perform One-Way ANOVA on the specified metric across provinces."""
  # Group data by Province
  grouped_data = [group[metric1].values for name, group in df.groupby(metric2)]

  # Perform One-Way ANOVA
  f_statistic, p_value = stats.f_oneway(*grouped_data)

  return f_statistic, p_value


def analyze_results(statistic, p_value, alpha=0.05):
  """Analyze the results of the t-test."""
  print(f'Statistic: {statistic}')
  print(f'P-value: {p_value}')
  if p_value < alpha:
    print("Reject the null hypothesis")
  else:
    print("Fail to reject the null hypothesis")
