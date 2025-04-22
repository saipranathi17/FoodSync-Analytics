# FoodSync-Analytics
A prototype for inventory management and sales prediction for small-scale Indian food businesses.

## Project Summary

This project implements a data analytics pipeline for a simulated food business with 30 inventory items: Milk Bread, Brown Bread, Multigrain Bread, Pav Bun, Plain Cake, Chilli Sauce, Rasgulla, Jalebi, Soy Sauce, Water, Chocolate Cake, Vanilla Cake, Black Forest Cake, Pineapple Cake, Red Velvet Cake, Milk, Curd, Paneer, Butter, Cheese, Samosa, Kachori, Dhokla, Vada Pav, Pani Puri, Cookies, Masala Puri, Samosa Chaat, Small Mixture, and Tomato Ketchup. The code performs the following:

- **Synthetic Data Generation**: Creates 90 days of simulated inventory and sales data, saved as `data/inventory_data.csv` and `data/sales_data.csv`. Includes realistic quantities (50–500 units for inventory, 50–200 units for sales with weekend boosts), expiration dates (1–60 days based on item perishability), and costs.
- **Exploratory Data Analysis (EDA)**: Analyzes sales trends for the top 5 items, visualizing daily sales in a line plot (`eda_sales_trends.png`) and providing summary statistics (e.g., mean, min, max sales per item).
- **Sales Prediction**: Uses an ARIMA model to forecast 7 days of sales for Rasgulla, visualized in `forecast_rasgulla.png` with historical and predicted values.
- **Alert System**: Generates alerts for low stock (quantity < 50 units) and expiring items (within 3 days) based on the latest inventory date.
- **Interactive Dashboard**: Creates Plotly charts showing sales trends for the top 5 items and current inventory levels, rendered interactively in a browser.

The code is designed for easy execution in Google Colab or a local Jupyter environment, producing reproducible outputs for data-driven inventory management.
The simulated plots for (`eda_sales_trends.png`) & `forecast_rasgulla.png`:
![image](https://github.com/user-attachments/assets/c835778c-11b8-4b56-ae02-04a33be29071)


## Setup
Run in Google Colab or install locally:
```bash
pip install pandas numpy matplotlib plotly statsmodels
