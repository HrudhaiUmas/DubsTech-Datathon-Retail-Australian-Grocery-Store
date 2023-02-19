"""
Machine Learning Linear Regression Model:
Current implementation is set up to find total sales
for Jan-March 2019, but can be modified to predict
other values like the quantity or total
manufacturing/purchasing costs of the store
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# load the data from csv
data = pd.read_csv('sales_data_2017_2018_data_updated.csv')

# filter the data to only include 2018
data_2018 = data[data['year'] == 2018]

# group the data by month and calculate the total selling price
# change 'total_selling_price' to another field if creating different predictions
monthly = data_2018.groupby('month_name')['total_selling_price'].sum()

# convert the monthly data to a dataframe
df = pd.DataFrame({'month': monthly.index, 'sales': monthly.values})

# convert the month names to numbers for the regression model
month_to_num = {
    'January': 1,
    'February': 2,
    'March': 3
}
df['month_num'] = df['month'].map(month_to_num)

# drop any rows with missing values
df.dropna(inplace=True)

# split the data into x and y
x = df[['month_num']]
y = df['sales']

# train the linear regression model
model = LinearRegression()
model.fit(x, y)

# predict the sales for January, February, and March of 2019
jan = model.predict([[1]])
feb = model.predict([[2]])
mar = model.predict([[3]])

# print the predicted sales for each month
print(f"Predicted sales for January 2019: ${jan[0]:,.2f}")
print(f"Predicted sales for February 2019: ${feb[0]:,.2f}")
print(f"Predicted sales for March 2019: ${mar[0]:,.2f}")

# calculate the accuracy of the model using R-squared and root mean squared error
y_pred = model.predict(x)
r_squared = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print(f"R-squared: {r_squared:.2f}")
print(f"Root mean squared error: {rmse:.2f}")

# plot the regression line
plt.plot(x, y, 'o')
plt.plot(x, model.predict(x))
plt.title('Predicted Monthly Sales (Jan-March 2019)')
plt.xlabel('Month')
plt.ylabel('Total Selling Price ($)')
plt.xticks([1, 2, 3], ['January', 'February', 'March'])
plt.show()