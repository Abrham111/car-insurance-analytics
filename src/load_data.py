import pandas as pd

def txt_to_csv(input_file, output_file):
  try:
    # Read the comma-delimited file into a DataFrame
    df = pd.read_csv(input_file, delimiter='|', skipinitialspace=True)

    # Save the DataFrame to a .csv file
    df.to_csv(output_file, index=False)

    print(f"Conversion complete. The .csv file is saved as {output_file}")
  except Exception as e:
    print(f"An error occurred: {e}")

def data_loader(file_path):
  data = pd.read_csv(file_path)
  return data