# Trying the predict the selling price

import pandas as pd 


# importing the data
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# -- checking for null values

#print(df.info())


# -- printing the top 10

#print(df.head(2))
#print(df.shape) # 4340 rows 8 columns

# finding out the unique things in the data 

#print([int(year) for year in sorted(df["year"].unique())])
#print(df["fuel"].unique())
#print(df["seller_type"].unique())
#print(df["transmission"].unique()) 
#print(df["owner"].unique())

# -- Checking for dublicate data

#print(df.duplicated().sum())
#dublicate_data = df[df.duplicated()]
#print(dublicate_data.head(30))

# learn ---- duplicate by default check the entire row (not just one column or 2) and if every column is identical tto the other it considers them as dublicate 
# if u want to check for dupliates only in some certain columsn then u have to use subset

# -- drouping the dublicate rows

df = df.drop_duplicates()
#print(df.duplicated().sum()) 
print(df.shape) # now its 3557,8

# checking for null values
#print(df.isna().sum())

# checking for strange numbers, like negative values

print(df[["year","selling_price","km_driven"]].describe())

# -- learn

# -- 