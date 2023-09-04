
import streamlit as st
import pandas as pd
import seaborn as sns 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 


def main():

 st.write("Upload a CSV or Excel file and explore its contents, perform statistical analysis with statsmodels.")
st.title("Data Exploration and Statsmodels App")

x_variables = []
y_variable = []


    # Create a file upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
if uploaded_file is not None:
        # Load data into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0)
        else:
            df = pd.read_excel(uploaded_file, index_col=0)

        # Display the first few rows of the DataFrame
        st.subheader("Data Preview (df.head())")
        st.write(df.head())

        # Show basic statistics of the data
        st.subheader("Data Statistics (df.describe())")
        st.write(df.describe())

        # Create dropdown menus to select X (independent) and y (dependent) variables
        st.sidebar.subheader("Select Variables")
        
        # Dropdown menu for X (independent variable)
        x_variables = st.sidebar.multiselect("Select Independent Variable (X)", df.columns)

        # Dropdown menu for y (dependent variable)
        y_variable = st.sidebar.selectbox("Select Dependent Variable (y)", df.columns)
        
if x_variables:
        # Fit a simple linear regression model
        X = df[x_variables]
        X = sm.add_constant(X)
        y = df[y_variable]

        model = sm.OLS(y, X).fit()

        # Display the model summary
        st.subheader("Linear Regression Model Summary")
        st.write(model.summary())

        # Create a scatter plot of the actual vs. predicted values
        # Create a scatter plot of the actual vs. predicted values
predicted_values = model.predict(X)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y, predicted_values, alpha=0.5)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs. Predicted Values")
st.pyplot(fig)  # Pass the Matplotlib figure to st.pyplot()

# Create a residual plot
residuals = y - predicted_values
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(predicted_values, residuals, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
st.pyplot(fig)  # Pass the Matplotlib figure to st.pyplot()

if __name__ == "__main__":
    main()





