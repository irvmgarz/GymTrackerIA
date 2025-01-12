import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor


#calculate IQR functions
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want to apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """
    dataset = dataset.copy()
    
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)

    # Convert to numpy arrays for element-wise operations
    low = low.to_numpy()
    high = high.to_numpy()

    prob = []
    mask = []

    # Pass all rows in the dataset
    for i in range(len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    
    # Add a column indicating outliers
    dataset[col + "_outlier"] = mask
    return dataset




# csv_file = "01_Data_Processed.csv"  # Replace with your CSV file path
#df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Pickle file
#pickle_file = "01_Data_Processed.pkl"  # Desired pickle file name
#df.to_pickle(pickle_file)


df= pd.read_pickle("01_Data_Processed.pkl")
outlier_columns= list(df.columns[1:7])
print(outlier_columns)
# Shortened column names for accelerometer
shortened_acc_columns = {'Accelerometer_x': 'acc_x', 'Accelerometer_y': 'acc_y', 'Accelerometer_z': 'acc_z'}
df_acc = df.rename(columns=shortened_acc_columns)

# Plot Accelerometer columns
accelerometer_columns = ['acc_x', 'acc_y', 'acc_z']
df_acc[accelerometer_columns + ['Label']].boxplot(by="Label", layout=(1, 3), figsize=(20, 10))
plt.suptitle("")  # Remove default title
plt.title("Boxplots for Accelerometer Data Grouped by Label")
plt.xlabel("Label")
plt.ylabel("Values")
#plt.show()  # Display the first set of boxplots

# Shortened column names for gyroscope
shortened_gyr_columns = {'Gyroscope_x': 'gyr_x', 'Gyroscope_y': 'gyr_y', 'Gyroscope_z': 'gyr_z'}
df_gyr = df.rename(columns=shortened_gyr_columns)

# Plot Gyroscope columns
gyroscope_columns = ['gyr_x', 'gyr_y', 'gyr_z']
df_gyr[gyroscope_columns + ['Label']].boxplot(by="Label", layout=(1, 3), figsize=(20, 10))
plt.suptitle("")  # Remove default title
plt.title("Boxplots for Gyroscope Data Grouped by Label")
plt.xlabel("Label")
plt.ylabel("Values")
#plt.show()  # Display the second set of boxplots

df= pd.read_pickle("01_Data_Processed.pkl")
outlier_columns= list(df.columns[1:7])
print(outlier_columns)
outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["Label"].unique():
        dataset= mark_outliers_chauvenet(df[df["Label"]==label], col)
        dataset.loc[dataset[col+"_outlier"],col]= np.nan
        outliers_removed_df.loc[(outliers_removed_df["Label"]== label), col]= dataset[col]
        n_outliers= len(df)- len(outliers_removed_df[col].dropna())
        print(f"Removed {n_outliers} from {col} for{label}")



outliers_removed_df.info()
outliers_removed_df.to_pickle("02_outliers_removed.pkl")