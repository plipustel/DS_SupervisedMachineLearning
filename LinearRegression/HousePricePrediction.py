import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('D:/TheCoder/DataScience/DS_SupervisedMachineLearning/datasets/AmesHousing.csv')

# --> STEP 1: Print all data or first 5
# 1a. Print all data:
# house = pd.DataFrame(data)

# # 1b. Print first 5 of data:
# print(data.head(5))

# # 1c. Print all columns:
# print(data.info())

# # --> STEP 2: EDA (Exploratory Data Analysis)
# # 2a. Check missing value
# print(f'Data null: {data.isnull().sum()}')

# # 2b. Summary statistics
# print(f'Summary statistics: {data.describe()}')


# # --> STEP 3: Visualize the data relationship:
# # 3a. Distribution Sale Price
# sns.histplot(data['SalePrice'], kde=True)
# plt.show()

# # 3b. Scatter plot between living area and SalePrice
# sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=data)
# plt.show()

# # --> STEP 4: Build Model using sklearn
# # 4a. Define feature target
# x = data.drop('SalePrice', axis=1)
# y = data['SalePrice']

# # 4b. Split into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# lr = LinearRegression()
# lr.fit(x_train, y_train)

# # 4c. Make prediction
# y_pred = lr.predict(x_test)


# # 4d. Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error {mse}')
# print(f'R^2 Score: {r2}')

x = data['Gr Liv Area'].values
y = data['SalePrice'].values

plt.scatter(x, y, color='green', label='test')
plt.xlabel('Sales Price')
plt.ylabel('Gr Liv Area')
plt.title('Linear Regression')

plt.show()