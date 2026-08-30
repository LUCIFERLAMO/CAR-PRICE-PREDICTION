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

#print(df[["year","selling_price","km_driven"]].describe())

# -- learn -- describe tells u if there r any negative numbers as u can see them in the min section

# --- selecting features for the model

# -- learn: what we r trying to predict we have to not put that in the model we can put the precious or the next values
# That would be like saying:   "Predict the price using the price."  That's called data leakage, and we'll avoid it.

# ---- sorting the data in assending order not required



#print(df.columns.to_list())

x = df[['year', 'km_driven', 'fuel', 'transmission', 'owner']]

#print(model_features.shape)


# -- storing the selling price in a variable which we will split for training and testing purpose

selling_pricee = df["selling_price"]
#print(selling_pricee)


# --- spliting the data for training (80/20)

split_data = int(len(x) * 0.8)
print(split_data)

# -- split 

x_train = x.iloc[:split_data]
x_test = x.iloc[split_data:]

# -- spliting the prediction values for training and testing 

y_price_train = selling_pricee.iloc[:split_data]
y_price_test = selling_pricee.iloc[split_data:]


# importing linear regression to make the prediction

from sklearn.linear_model import LinearRegression

#sell_price = LinearRegression()

# train 

#sell_price.fit(x_train,y_price_train)

# predict

#ans = sell_price.predict(x_test)

# -- compare the error

from sklearn.metrics import mean_absolute_error

#diff = mean_absolute_error(ans,y_price_test)

#print(diff)

# -- lesson -- did not work as kinear regression needed numerical data but we hvae some catagircal data so in order to slove tbis we need to do one hot encoding
# one hot encoding means lets say wr have petrol cng diseal in fuel category 
# now when we do one hot encoding we convert it into petrol as 001 diseal as 010 so its more like a yes or no 

# fuel = ["Petrol", "Diesel", "CNG", "Petrol"]

#One-hot encoding turns it into:

#fuel_CNG    fuel_Diesel    fuel_Petrol
#    0            0              1
#    0            1              0
#    1            0              0
#    0            0              1

#Each column answers a simple yes/no question:

#fuel_CNG     → Is this car CNG?
#fuel_Diesel  → Is this car Diesel?
#fuel_Petrol  → Is this car Petrol?

#1 means yes.

#0 means no.

fuel_encoded = pd.get_dummies(df["fuel"],drop_first=True) # dummies automatically sperates the unique categories from the fuel column and assigns values to it like true and false

# drop_first=True remove one column  from the existing ones 
# ex: if we  hv CNG Diesel Electric LPG Petrol and if petrol is removed
# then if all the 4 remaing category is 0 then the model understands it is petrol 
# if its 1 0 0 0 then the model undersrands it is CNG
# the the column whihc is gona be removed depends on the pandas category ordering


print(fuel_encoded.head(10))

# we will do the same for the remaiing category labels

# done for the day