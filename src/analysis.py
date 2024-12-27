import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_variability(df, columns_of_interest):
  """
  Calculate mean, variance, standard deviation, and coefficient of variation
  for specified numerical columns in a DataFrame.

  Parameters:
  df (pd.DataFrame): The DataFrame containing the data.
  columns_of_interest (list): List of column names for which to calculate variability.

  Returns:
  dict: A dictionary containing the calculated statistics for each column.
  """
  results = {}

  for column in columns_of_interest:
    mean = df[column].mean()
    variance = df[column].var()
    std_dev = df[column].std()
    cv = (std_dev / mean) * 100 if mean != 0 else None  # Avoid division by zero

    results[column] = {
      'Mean': mean,
      'Variance': variance,
      'Standard Deviation': std_dev,
      'Coefficient of Variation (%)': cv
    }

  return results

def plot_distributions(df):
  """Plot histograms for numerical columns and bar charts for categorical columns."""
  numerical_cols = df.select_dtypes(include=['number']).columns
  categorical_cols = df.select_dtypes(include=['object']).columns

  # Plot histograms for numerical columns
  for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

  # Plot bar charts for categorical columns
  for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


def plot_correlations(df, x_col, y_col, zip_col):
  """Explore relationships between the monthly changes in TotalPremium and TotalClaims as a function of PostalCode."""
  # Limit the number of unique values in zip_col for better visualization
  top_zip_codes = df[zip_col].value_counts().index[:10]
  df[zip_col] = df[zip_col].apply(lambda x: x if x in top_zip_codes else 'Other')

  # Filter data
  df_filtered = df[df[zip_col] != 'Other'].dropna(subset=[x_col, y_col, zip_col])
  print(f"Filtered DataFrame shape: {df_filtered.shape}")

  # Sampling for large datasets
  if len(df_filtered) > 5000:
    df_filtered = df_filtered.sample(n=5000, random_state=42)
    print(f"Sampled DataFrame shape: {df_filtered.shape}")

  # Scatter plot
  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df_filtered, x=x_col, y=y_col, hue=zip_col, legend='brief')
  plt.title(f'Scatter Plot of {x_col} vs {y_col} by {zip_col}')
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  plt.legend(title=zip_col)
  plt.show()

  # Correlation matrix
  correlation_matrix = df[[x_col, y_col]].corr()
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation Matrix')
  plt.show()


def compare_trends(df, group_col, value_col):
  """Compare the change in a specified column across different groups."""
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=df, x=group_col, y=value_col, estimator='mean', ci=None)
  plt.title(f'Trends in {value_col} by {group_col}')
  plt.xlabel(group_col)
  plt.ylabel(value_col)
  plt.xticks(rotation=45)
  plt.show()

def detect_outliers(df, col):
  """Use box plots to detect outliers in a numerical column."""
  # Check if the column exists
  if col not in df.columns:
    print(f"Column '{col}' not found in DataFrame.")
    return

  # Remove NaN values from the column
  df_filtered = df[df[col].notna()]

  # Ensure the column is numeric
  if not pd.api.types.is_numeric_dtype(df_filtered[col]):
    print(f"Column '{col}' is not numeric.")
    return

  # Debugging: Print the number of valid entries and the filtered data
  print(f"Valid entries for '{col}': {len(df_filtered)}")
  print(df_filtered[col].dropna())

  # Create the box plot
  plt.figure(figsize=(8, 4))
  sns.boxplot(x=df_filtered[col])
  plt.title(f'Box Plot of {col}')
  plt.xlabel(col)
  plt.show()

def produce_visualizations(df):
  """Produce three creative plots that capture key insights."""
  if 'TotalPremium' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['TotalPremium'], bins=10, kde=True)
    plt.title('Distribution of Total Premium')
    plt.show()
  else:
    print("Column 'TotalPremium' not found in DataFrame.")

  total_claim_col = 'TotalClaims'  # Ensure this matches your DataFrame
  if total_claim_col in df.columns:
    # Remove NaN values for plotting
    df = df.dropna(subset=[total_claim_col])
    if pd.api.types.is_numeric_dtype(df[total_claim_col]):
      plt.figure(figsize=(8, 4))
      sns.boxplot(x=df[total_claim_col])
      plt.title('Box Plot of Total Claims')
      plt.xlabel(total_claim_col)
      plt.show()
    else:
      print(f"Column '{total_claim_col}' is not numeric.")
  else:
    print(f"Column '{total_claim_col}' not found in DataFrame.")

  if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', hue='PostalCode')
    plt.title('Total Premium vs Total Claim by Zip Code')
    plt.show()
  else:
    print("One or both columns 'TotalPremium' and 'TotalClaims' not found in DataFrame.")
