import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor


csv_file = "../../data/01_Data_Processed.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Pickle file
pickle_file = "01_Data_Processed.pkl"  # Desired pickle file name
df.to_pickle(pickle_file)

pickle_file = "02_outliers_removed.pkl"  # Desired pickle file name
df.to_pickle(pickle_file)

df= pd.read_pickle("01_Data_Processed.pkl")
df2= pd.read_pickle("02_outliers_removed.pkl")
df.info()
df2.info()
df["Set"] = pd.to_numeric(df["Set"], errors="coerce")
set_df = df[df["Set"] == 1]
print(f"Number of rows in filtered DataFrame (Set == 1): {len(set_df)}")
print(set_df.head())

plt.plot(set_df["Accelerometer_y"].reset_index(drop=True))
plt.show()