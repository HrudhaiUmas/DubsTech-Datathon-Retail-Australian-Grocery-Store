import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# load the data from csv
data = pd.read_csv('sales_data_2017_2018_data_updated.csv')

# filter the data to only include 2018
data_2018 = data[data['year'] == 2018]

# group the data by month and calculate the total selling price
monthly_sales = data_2018.groupby('month_name')['total_selling_price'].sum()

# convert the monthly sales data to a dataframe
df = pd.DataFrame({'month': monthly_sales.index, 'sales': monthly_sales.values})

# convert the month names to numbers for the regression model
month_to_num = {
    'January': 1,
    'February': 2,
    'March': 3
}
df['month_num'] = df['month'].map(month_to_num)

# drop any rows with missing values
df.dropna(inplace=True)

# split the data into X and y
X = df[['month_num']]
y = df['sales']

# train the linear regression model
model = LinearRegression()
model.fit(X, y)

# predict the sales for January, February, and March of 2019
jan_sales = model.predict([[1]])
feb_sales = model.predict([[2]])
mar_sales = model.predict([[3]])

# print the predicted sales for each month
print(f"Predicted sales for January 2019: ${jan_sales[0]:,.2f}")
print(f"Predicted sales for February 2019: ${feb_sales[0]:,.2f}")
print(f"Predicted sales for March 2019: ${mar_sales[0]:,.2f}")

# calculate the accuracy of the model using R-squared and root mean squared error
y_pred = model.predict(X)
r_squared = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print(f"R-squared: {r_squared:.2f}")
print(f"Root mean squared error: {rmse:.2f}")

# plot the regression line and the actual sales data
plt.plot(X, y, 'o')
plt.plot(X, model.predict(X))
plt.title('Predicted Monthly Sales (Jan-March 2019)')
plt.xlabel('Month')
plt.ylabel('Total Selling Price ($)')
plt.xticks([1, 2, 3], ['January', 'February', 'March'])
plt.show()