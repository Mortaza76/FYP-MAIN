import pandas as pd
from AutoClean import AutoClean

# Step 1: Load the dataset
df = pd.read_csv("/Users/ameermortaza/Desktop/amazon.csv")
print("Original Dataset:")
print(df)

# Step 2: Instantiate AutoClean to clean the dataset
autoclean = AutoClean(
    df,
    mode="auto",            # Automatically handle data cleaning
    missing_num="mean",     # Fill missing numeric values with the mean
    missing_categ="mode",   # Fill missing categorical values with the mode
    encode_categ="onehot",  # Apply one-hot encoding to categorical variables
    outliers="drop"         # Drop rows with outliers
)

# Step 3: Access the cleaned dataset
cleaned_df = autoclean.output

# Step 4: Display the cleaned dataset
print("\nCleaned Dataset:")
print(cleaned_df)

# Step 5: Save the cleaned dataset to a CSV file
cleaned_df.to_csv("cleaned_test.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_test.csv'.")
