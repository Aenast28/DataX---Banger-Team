# %%
# Import libraries
from catboost import CatBoostRegressor
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# Deprecated function warning is disabled
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data : put the file in the same directory as the script and use relative path to the data file (Tested Using github repository)
csv_file="Data\\interim\\features -1-nodum.csv"


data = pd.read_csv(csv_file)
data1 =pd.read_csv("Data\\interim\\features-j-2.csv")
data2 =pd.read_csv("Data\\interim\\features_cal_cal.csv")
data3 =pd.read_csv("Data\\interim\\features_cal_rev.csv")
merged_df = pd.merge(data2, data1, on='listing_id', how='left')
merged_df = pd.merge(merged_df, data, on='listing_id', how='left')
merged_df = pd.merge(merged_df, data3, on='listing_id', how='left')
merged_df = merged_df.astype({col: 'str' for col in merged_df.select_dtypes(include='object').columns})
merged_df = merged_df.sort_values(by='date', ascending=False)
mode_value = merged_df['Verification'].mode()[0]
merged_df['Verification'].fillna(mode_value, inplace=True)
mode_value = merged_df['Distance'].mode()[0]
merged_df['Distance'].fillna(mode_value, inplace=True)
merged_df['sentiment_score'].fillna(0.5, inplace=True)
column_names = merged_df.columns.tolist()

# Drop rows with NaN values
merged_df = merged_df.dropna()

# Get column names
column_names = merged_df.columns

# %%
# Author text
st.sidebar.markdown('<h5 style="color: black;"> Powered by Cats, Boosting and a power of friendship </h5>', unsafe_allow_html=True)

# Sidebar for user input selection
st.sidebar.markdown('<h1 style="color: blue;">Select One output and at least one input Variable</h1>', unsafe_allow_html=True)
# Select output variable
output_variable_model = st.sidebar.selectbox('Select One output Variable', column_names)

# Select input variables to predict the target variable (output)
input_variables_model = st.sidebar.multiselect('Select at least one input Variable', column_names, default=['maximum_nights',
  'new_year',
  'bathrooms_numeric',
  'amenities_num',
  'host_response_time',
  'ranking',
  'Distance',"host_is_superhost","Verification","instant_bookable","has_availability","neighbourhood_cleansed","room_type"])

if not output_variable_model or not input_variables_model:
    st.warning('Select One output and at least one input Variable to start.')

# User option for setting the rate of test data
test_data_rate = st.sidebar.slider('Select the rate of test data (%)', 0, 100, 20, 1)

# %%
# Define input features (X) and target variable (y) for model training
categorical_features=["host_is_superhost","Verification","instant_bookable","has_availability","neighbourhood_cleansed","room_type"]
x_train = merged_df.iloc[-7000:][input_variables_model]
y_train = merged_df.iloc[-7000:][output_variable_model]
x_test = merged_df.drop(x_train.index)[input_variables_model]

y_test = merged_df.drop(y_train.index)[output_variable_model]

# Train Random Forest model
model= CatBoostRegressor(eval_metric='R2',
                              iterations = 190, 
                              cat_features = categorical_features, 
                              random_state = 123
                              )
model.fit(x_train, y_train)

# %%
# Streamlit application
# Model Training and Validation

st.title('Machine Learning Model: Training, Validation and Prediction using CatBoost Algorithm')
st.markdown('<h4 style="color: black;"> The main use of this Streamlit app is to see the significance of the CatBoost features. </h4>', unsafe_allow_html=True)

# Display information about the trained model
st.header('Model Information')
st.write(f'Output Variable (Target): {output_variable_model}')
st.write(f'Input Variables: {", ".join(input_variables_model)}')
st.write(f'Training Data Shape: {x_train.shape}')
st.write(f'Test Data Shape: {x_test.shape}')

# Display scatter plot chart of predicted vs observed values for test data
st.subheader('Scatter Plot: Predicted vs Observed (Test Data)')

test_predictions = model.predict(x_test)
scatter_df = pd.DataFrame({'Observed': y_test, 'Predicted': test_predictions})
fig, ax = plt.subplots(figsize=(8, 5))

# Create a scatter plot with a regression line
fig, ax = plt.subplots(figsize=(8, 5))
scatter_plot = sns.regplot(x='Observed', y='Predicted', data=scatter_df, ax=ax)

# Calculate R-squared value
r_squared = r2_score(y_test, test_predictions)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

# Add R-squared and RMSE as annotations
text = f'R-squared: {r_squared:.2f}\nRMSE: {rmse:.2f}'
ax.text(0.05, 0.9, text, transform=ax.transAxes, color='blue', fontsize=12)

# Customize title and labels
scatter_plot.set_title(f'Predicted vs Observed {output_variable_model}', color='blue')
scatter_plot.set_xlabel('Observed', color='blue')
scatter_plot.set_ylabel('Predicted', color='blue')

# Show the plot
st.pyplot(fig)

# Display feature importance chart
st.subheader('Feature Importance Chart')
feature_importance_model = pd.Series(model.feature_importances_, index=input_variables_model).sort_values(ascending=False)
fig, ax=plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importance_model, y=feature_importance_model.index, palette='viridis')
ax.set_title('CatBoost Feature Importance')
st.pyplot(fig)

# %%
# Use the model for prediction
st.title('Use the Model for Prediction')
st.markdown('<h4 style="color: black;"> Use Sidebar menu to select the values of input variables to predict the target variable. </h4>', unsafe_allow_html=True)

# User input for feature values
st.sidebar.markdown('<h2 style="color: blue;"> Select the values of input variables to predict the target variable</h2>', unsafe_allow_html=True)
user_input_prediction = {}
from sklearn.preprocessing import LabelEncoder


for column in input_variables_model:
    try:
        user_input_prediction[column] = st.sidebar.slider(f'Select {column}', float(merged_df[column].min()), float(merged_df[column].max()), float(merged_df[column].mean()))
    except ValueError:
        user_input_prediction[column] = 1

# Predict and display result
prediction = model.predict(pd.DataFrame([user_input_prediction]))
st.subheader('Prediction')
st.write(f'The predicted {output_variable_model} value is: {prediction[0]:.5f}')

# Display a bar chart for the predicted output
st.subheader('Predicted Output Chart')
prediction_data = pd.DataFrame({output_variable_model: [prediction[0]]})
fig, ax=plt.subplots(figsize=(8, 5))
sns.barplot(data=prediction_data, palette=['orange'])
ax.set_title(f'Predicted {output_variable_model} Value')
ax.set_ylabel('Value')
st.pyplot(fig)

# %%



