import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== SALES FORECASTING ANALYSIS ===\n")

# 1. Prepare time series data
print("1. TIME SERIES DATA PREPARATION")
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

# Aggregate sales by month for forecasting
monthly_sales = df.groupby(pd.Grouper(key='ORDERDATE', freq='ME'))['SALES'].sum().reset_index()
monthly_sales.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' columns

print(f"Monthly sales data: {len(monthly_sales)} months")
print(f"Date range: {monthly_sales['ds'].min()} to {monthly_sales['ds'].max()}")
print(f"Total sales in period: ${monthly_sales['y'].sum():,.2f}")
print()

# 2. Moving Average Forecasting
print("2. MOVING AVERAGE FORECASTING")

# Calculate different moving averages
monthly_sales['MA_3'] = monthly_sales['y'].rolling(window=3).mean()
monthly_sales['MA_6'] = monthly_sales['y'].rolling(window=6).mean()
monthly_sales['MA_12'] = monthly_sales['y'].rolling(window=12).mean()

# Simple forecasting using last MA value
last_ma_3 = monthly_sales['MA_3'].iloc[-1]
last_ma_6 = monthly_sales['MA_6'].iloc[-1]
last_ma_12 = monthly_sales['MA_12'].iloc[-1]

print("Moving Average Forecasts (next month):")
print(f"3-month MA: ${last_ma_3:,.2f}")
print(f"6-month MA: ${last_ma_6:,.2f}")
print(f"12-month MA: ${last_ma_12:,.2f}")
print()

# Calculate accuracy metrics for historical data
actual = monthly_sales['y'].iloc[12:]  # Start from month 13 for 12-month MA comparison
predicted_ma_12 = monthly_sales['MA_12'].iloc[12:]

if len(actual) > 0 and len(predicted_ma_12) > 0:
    mae_12 = mean_absolute_error(actual, predicted_ma_12)
    rmse_12 = np.sqrt(mean_squared_error(actual, predicted_ma_12))
    mape_12 = np.mean(np.abs((actual - predicted_ma_12) / actual)) * 100

    print("12-Month Moving Average Accuracy:")
    print(f"MAE: ${mae_12:,.2f}")
    print(f"RMSE: ${rmse_12:,.2f}")
    print(".2f")
    print()

# 3. Prophet Model Forecasting
print("3. PROPHET MODEL FORECASTING")

# Initialize and fit Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)

model.fit(monthly_sales[['ds', 'y']])

# Create future dates for forecasting (next 12 months)
future_dates = model.make_future_dataframe(periods=12, freq='ME')

# Generate forecasts
prophet_forecast_df = model.predict(future_dates)

# Extract forecast components
forecast_next_month = prophet_forecast_df[prophet_forecast_df['ds'] > monthly_sales['ds'].max()].iloc[0]
prophet_forecast = forecast_next_month['yhat']

print("Prophet Forecast (next month):")
print(f"Predicted sales: ${prophet_forecast:,.2f}")
print(f"Lower bound: ${forecast_next_month['yhat_lower']:,.2f}")
print(f"Upper bound: ${forecast_next_month['yhat_upper']:,.2f}")
print()

# Calculate Prophet accuracy on historical data
historical_forecast = prophet_forecast_df[prophet_forecast_df['ds'] <= monthly_sales['ds'].max()]
actual_historical = monthly_sales['y']

if len(historical_forecast) == len(actual_historical):
    prophet_mae = mean_absolute_error(actual_historical, historical_forecast['yhat'])
    prophet_rmse = np.sqrt(mean_squared_error(actual_historical, historical_forecast['yhat']))
    prophet_mape = np.mean(np.abs((actual_historical - historical_forecast['yhat']) / actual_historical)) * 100

    print("Prophet Model Accuracy (Historical):")
    print(f"MAE: ${prophet_mae:,.2f}")
    print(f"RMSE: ${prophet_rmse:,.2f}")
    print(".2f")
    print()

# 4. Forecast Comparison
print("4. FORECAST METHOD COMPARISON")
forecast_comparison = pd.DataFrame({
    'Method': ['3-Month MA', '6-Month MA', '12-Month MA', 'Prophet'],
    'Next_Month_Forecast': [last_ma_3, last_ma_6, last_ma_12, prophet_forecast]
})

print("Next Month Sales Forecasts:")
for _, row in forecast_comparison.iterrows():
    print(f"{row['Method']}: ${row['Next_Month_Forecast']:,.2f}")
print()

# 5. Create forecasting visualizations
print("5. FORECAST VISUALIZATIONS")

# Plot 1: Historical sales with moving averages
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
plt.plot(monthly_sales['ds'], monthly_sales['y'], label='Actual Sales', color='blue', linewidth=2)
plt.plot(monthly_sales['ds'], monthly_sales['MA_3'], label='3-Month MA', color='orange', linestyle='--')
plt.plot(monthly_sales['ds'], monthly_sales['MA_6'], label='6-Month MA', color='green', linestyle='--')
plt.plot(monthly_sales['ds'], monthly_sales['MA_12'], label='12-Month MA', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.title('Historical Sales with Moving Averages')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot 2: Prophet forecast
plt.subplot(2, 2, 2)
plt.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Sales', color='blue', linewidth=2)
plt.plot(prophet_forecast_df['ds'], prophet_forecast_df['yhat'], label='Prophet Forecast', color='red', linewidth=2)
plt.fill_between(prophet_forecast_df['ds'], prophet_forecast_df['yhat_lower'], prophet_forecast_df['yhat_upper'],
                color='red', alpha=0.2, label='Confidence Interval')
plt.axvline(x=monthly_sales['ds'].max(), color='black', linestyle='--', alpha=0.7, label='Forecast Start')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.title('Prophet Sales Forecast')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Plot 3: Forecast comparison
plt.subplot(2, 2, 3)
methods = forecast_comparison['Method']
forecasts = forecast_comparison['Next_Month_Forecast']
bars = plt.bar(methods, forecasts, color=['orange', 'green', 'red', 'purple'])
plt.ylabel('Forecasted Sales ($)')
plt.title('Next Month Forecast Comparison')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

# Add value labels on bars
for bar, forecast in zip(bars, forecasts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(forecasts)*0.01,
            '.0f', ha='center', va='bottom')

# Plot 4: Forecast accuracy comparison (if available)
plt.subplot(2, 2, 4)
if 'mae_12' in locals() and 'prophet_mae' in locals():
    methods_acc = ['12-Month MA', 'Prophet']
    mae_values = [mae_12, prophet_mae]
    bars_acc = plt.bar(methods_acc, mae_values, color=['red', 'purple'])
    plt.ylabel('Mean Absolute Error ($)')
    plt.title('Forecast Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    for bar, mae in zip(bars_acc, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                '.0f', ha='center', va='bottom')
else:
    plt.text(0.5, 0.5, 'Accuracy metrics\nnot available\nfor all methods', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Forecast Accuracy (Data Limited)')

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/sales_forecasting.png', dpi=300, bbox_inches='tight')
plt.close()

print("Forecasting visualizations saved to sales_forecasting.png")

# 6. Export forecast results
forecast_future = prophet_forecast_df[prophet_forecast_df['ds'] > monthly_sales['ds'].max()]
if len(forecast_future) > 0:
    forecast_results = forecast_future.head(12).copy()
    forecast_results = forecast_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_results.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
    forecast_results.to_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_forecast_prophet.csv', index=False)
    print("Prophet forecast results exported to sales_forecast_prophet.csv")
else:
    print("No future forecast data available")

# 7. Forecasting insights and recommendations
print("\n=== FORECASTING INSIGHTS & RECOMMENDATIONS ===")

# Trend analysis
last_12_months = monthly_sales['y'].tail(12)
avg_last_12 = last_12_months.mean()
avg_first_12 = monthly_sales['y'].head(12).mean()
trend_pct = ((avg_last_12 - avg_first_12) / avg_first_12) * 100

print("Trend Analysis:")
print(".1f")
print(f"Average of last 12 months: ${avg_last_12:,.2f}")
print(f"Average of first 12 months: ${avg_first_12:,.2f}")
print()

# Forecast reliability assessment
if 'prophet_mape' in locals():
    if prophet_mape < 10:
        reliability = "High"
    elif prophet_mape < 20:
        reliability = "Moderate"
    else:
        reliability = "Low"

    print(f"Forecast Reliability: {reliability} (MAPE: {prophet_mape:.1f}%)")

print()

# Business recommendations based on forecasts
if len(forecast_future) > 0:
    avg_forecast = forecast_future['yhat'].head(12).mean()
    current_avg = monthly_sales['y'].tail(6).mean()

    print("Business Recommendations:")
    if avg_forecast > current_avg * 1.1:
        print("- Positive growth forecast: Prepare for increased demand")
        print("- Consider inventory expansion and staffing increases")
    elif avg_forecast < current_avg * 0.9:
        print("- Declining forecast: Implement cost control measures")
        print("- Focus on customer retention and new customer acquisition")
    else:
        print("- Stable forecast: Maintain current operations")
        print("- Focus on efficiency improvements and market share growth")

    print(f"- Current 6-month average: ${current_avg:,.2f}")
    print(f"- Forecast 12-month average: ${avg_forecast:,.2f}")
    if len(forecast_future) >= 12:
        print(f"- Forecast range: ${forecast_future['yhat_lower'].head(12).mean():,.2f} - ${forecast_future['yhat_upper'].head(12).mean():,.2f}")
else:
    print("Business Recommendations: Insufficient forecast data")

# Seasonal patterns
monthly_sales['Month'] = monthly_sales['ds'].dt.month
monthly_avg = monthly_sales.groupby('Month')['y'].mean()
best_month = monthly_avg.idxmax()
worst_month = monthly_avg.idxmin()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"- Best performing month (historical): {month_names[best_month-1]} (${monthly_avg.max():,.2f})")
print(f"- Worst performing month (historical): {month_names[worst_month-1]} (${monthly_avg.min():,.2f})")