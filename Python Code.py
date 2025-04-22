import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings("ignore")

# 1. Generate Synthetic Data with Provided 30 Items
def generate_synthetic_data():
    # Create date range (90 days)
    start_date = datetime(2025, 1, 22)
    dates = [start_date + timedelta(days=x) for x in range(90)]
    
    # Provided inventory items (exactly as listed)
    items = [
        'Milk Bread', 'Brown Bread', 'Multigrain Bread', 'Pav Bun', 'Plain Cake',
        'Chilli Sauce', 'Rasgulla', 'Jalebi', 'Soy Sauce', 'Water',
        'Chocolate Cake', 'Vanilla Cake', 'Black Forest Cake', 'Pineapple Cake', 'Red Velvet Cake',
        'Milk', 'Curd', 'Paneer', 'Butter', 'Cheese',
        'Samosa', 'Kachori', 'Dhokla', 'Vada Pav', 'Pani Puri',
        'Cookies', 'Masala Puri', 'Samosa Chaat', 'Small Mixture', 'Tomato Ketchup'
    ]
    
    # Generate inventory data
    inventory_data = {
        'Item': [],
        'Quantity': [],
        'Expiration_Date': [],
        'Cost_Per_Unit': [],
        'Date': []
    }
    
    # Generate sales data
    sales_data = {
        'Date': [],
        'Item': [],
        'Sales': []
    }
    
    for date in dates:
        for item in items:
            # Inventory: Quantities (50–500 units)
            quantity = np.random.randint(50, 500)
            # Expiration: 1–3 days for perishables, 7–14 days for others
            if item in ['Milk', 'Curd', 'Paneer', 'Butter', 'Cheese', 'Rasgulla', 'Jalebi']:
                expiration_days = np.random.randint(1, 4)
            elif item in ['Water', 'Chilli Sauce', 'Soy Sauce', 'Tomato Ketchup']:
                expiration_days = np.random.randint(30, 60)  # Longer shelf life
            else:
                expiration_days = np.random.randint(7, 15)
            expiration = date + timedelta(days=expiration_days)
            # Cost: Realistic pricing for Indian market
            if 'Bread' in item or item == 'Pav Bun':
                cost = 20
            elif 'Cake' in item:
                cost = 150
            elif item in ['Milk', 'Curd', 'Paneer', 'Butter', 'Cheese']:
                cost = 45
            elif item in ['Rasgulla', 'Jalebi']:
                cost = 35
            elif item in ['Samosa', 'Kachori', 'Dhokla', 'Vada Pav', 'Pani Puri', 'Masala Puri', 'Samosa Chaat']:
                cost = 15
            else:  # Cookies, Small Mixture, Sauces, Water
                cost = 10
            
            inventory_data['Item'].append(item)
            inventory_data['Quantity'].append(quantity)
            inventory_data['Expiration_Date'].append(expiration)
            inventory_data['Cost_Per_Unit'].append(cost)
            inventory_data['Date'].append(date)
            
            # Sales: Quantities (50–200 units/day, weekend boost)
            base_sales = np.random.randint(50, 150)
            if date.weekday() >= 5:  # Weekend boost
                base_sales *= 1.5
            sales_data['Date'].append(date)
            sales_data['Item'].append(item)
            sales_data['Sales'].append(int(base_sales))
    
    inventory_df = pd.DataFrame(inventory_data)
    sales_df = pd.DataFrame(sales_data)
    
    # Save to CSV for GitHub
    os.makedirs('data', exist_ok=True)
    inventory_df.to_csv('data/inventory_data.csv', index=False)
    sales_df.to_csv('data/sales_data.csv', index=False)
    
    return inventory_df, sales_df

# 2. Exploratory Data Analysis (EDA)
def perform_eda(sales_df):
    print("=== EDA: Sales Trends ===")
    # Aggregate sales by date for top 5 items (to avoid cluttered plot)
    top_items = sales_df.groupby('Item')['Sales'].sum().nlargest(5).index
    sales_pivot = sales_df[sales_df['Item'].isin(top_items)].pivot(index='Date', columns='Item', values='Sales')
    
    # Plot sales trends
    plt.figure(figsize=(12, 6))
    for item in sales_pivot.columns:
        plt.plot(sales_pivot.index, sales_pivot[item], label=item)
    plt.title('Daily Sales Trends (Top 5 Items)')
    plt.xlabel('Date')
    plt.ylabel('Sales (Units)')
    plt.legend()
    plt.grid(True)
    plt.savefig('eda_sales_trends.png')
    plt.show()
    
    # Summary statistics
    print("\nSales Summary Statistics:")
    print(sales_df.groupby('Item')['Sales'].describe())

# 3. Sales Prediction with ARIMA
def predict_sales(sales_df, item, forecast_days=7):
    print(f"\n=== Sales Prediction for {item} ===")
    # Filter sales for the item
    item_sales = sales_df[sales_df['Item'] == item][['Date', 'Sales']].set_index('Date')
    item_sales = item_sales.resample('D').sum().fillna(0)
    
    # Train ARIMA model
    model = ARIMA(item_sales['Sales'], order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast next 7 days
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_dates = [item_sales.index[-1] + timedelta(days=x) for x in range(1, forecast_days + 1)]
    
    # Plot historical and forecasted sales
    plt.figure(figsize=(12, 6))
    plt.plot(item_sales.index, item_sales['Sales'], label='Historical Sales')
    plt.plot(forecast_dates, forecast, label='Forecasted Sales', linestyle='--')
    plt.title(f'Sales Forecast for {item}')
    plt.xlabel('Date')
    plt.ylabel('Sales (Units)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'forecast_{item.lower().replace(" ", "_")}.png')
    plt.show()
    
    # Return forecast
    return pd.DataFrame({'Date': forecast_dates, 'Forecasted_Sales': forecast})

# 4. Alert System
def generate_alerts(inventory_df, current_date):
    print("\n=== Alerts ===")
    inventory_today = inventory_df[inventory_df['Date'] == current_date]
    
    for _, row in inventory_today.iterrows():
        item = row['Item']
        quantity = row['Quantity']
        expiration = row['Expiration_Date']
        
        # Low stock alert (less than 10% of max quantity ~50)
        if quantity < 50:
            print(f"Low Stock Alert: Only {quantity} units of {item} remaining!")
        
        # Expiration alert (within 3 days)
        days_to_expire = (expiration - current_date).days
        if days_to_expire <= 3:
            print(f"Expiration Alert: {item} expires in {days_to_expire} days on {expiration.date()}!")

# 5. Basic Dashboard with Plotly
def create_dashboard(sales_df, inventory_df, current_date):
    print("\n=== Dashboard ===")
    # Sales trend for top 5 items
    top_items = sales_df.groupby('Item')['Sales'].sum().nlargest(5).index
    sales_pivot = sales_df[sales_df['Item'].isin(top_items)].pivot(index='Date', columns='Item', values='Sales')
    fig1 = px.line(sales_pivot, title='Sales Trends (Top 5 Items)')
    fig1.update_layout({'xaxis_title': 'Date', 'yaxis_title': 'Sales (Units)'})
    fig1.update_layout(showlegend=True)
    fig1.update_xaxes(rangeslider_visible=True)
    fig1.show()
    
    # Inventory status
    inventory_today = inventory_df[inventory_df['Date'] == current_date]
    fig2 = px.bar(inventory_today, x='Item', y='Quantity', title=f'Inventory Levels on {current_date.date()}')
    fig2.update_layout(xaxis_tickangle=45)
    fig2.show()

# Main Execution
if __name__ == "__main__":
    # Generate synthetic data
    inventory_df, sales_df = generate_synthetic_data()
    
    # Perform EDA
    perform_eda(sales_df)
    
    # Predict sales for Rasgulla (popular Indian sweet)
    forecast_df = predict_sales(sales_df, 'Rasgulla')
    print("\nSales Forecast for Rasgulla:")
    print(forecast_df)
    
    # Generate alerts for the last date
    current_date = inventory_df['Date'].max()
    generate_alerts(inventory_df, current_date)
    
    # Create dashboard
    create_dashboard(sales_df, inventory_df, current_date)
