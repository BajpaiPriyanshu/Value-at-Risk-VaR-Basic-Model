import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]
    initial_prices = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 330, 'TSLA': 1000}
    stock_data = {}
    for stock, initial_price in initial_prices.items():
        daily_returns = np.random.normal(0.0008, 0.02, len(dates))
        prices = [initial_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        stock_data[stock] = prices[1:]
    df = pd.DataFrame(stock_data, index=dates)
    return df

class VaRCalculator:
    def __init__(self, price_data):
        self.prices = price_data.copy()
        self.returns = self.calculate_returns()
        
    def calculate_returns(self):
        returns = self.prices.pct_change().dropna()
        return returns
    
    def calculate_var_single_stock(self, stock, confidence_level=0.95, time_horizon=1):
        stock_returns = self.returns[stock].values
        sorted_returns = np.sort(stock_returns)
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(sorted_returns, percentile)
        var_scaled = var_return * np.sqrt(time_horizon)
        return var_return, var_scaled
    
    def calculate_portfolio_var(self, weights, confidence_level=0.95, time_horizon=1):
        portfolio_returns = (self.returns * weights).sum(axis=1)
        sorted_portfolio_returns = np.sort(portfolio_returns.values)
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(sorted_portfolio_returns, percentile)
        var_scaled = var_return * np.sqrt(time_horizon)
        return var_return, var_scaled, portfolio_returns
    
    def get_worst_case_scenarios(self, stock, n_scenarios=5):
        stock_returns = self.returns[stock]
        worst_scenarios = stock_returns.nsmallest(n_scenarios)
        return worst_scenarios
    
    def create_var_report(self, confidence_levels=[0.90, 0.95, 0.99], time_horizons=[1, 5]):
        print("=" * 80)
        print("VALUE AT RISK (VaR) REPORT - HISTORICAL METHOD")
        print("=" * 80)
        print()
        print("1. INDIVIDUAL STOCK VaR ANALYSIS")
        print("-" * 50)
        for stock in self.prices.columns:
            print(f"\n{stock} Stock Analysis:")
            print(f"Current Price: ${self.prices[stock].iloc[-1]:.2f}")
            print(f"Average Daily Return: {self.returns[stock].mean():.4f} ({self.returns[stock].mean()*100:.2f}%)")
            print(f"Daily Volatility: {self.returns[stock].std():.4f} ({self.returns[stock].std()*100:.2f}%)")
            for confidence in confidence_levels:
                for horizon in time_horizons:
                    var_1day, var_scaled = self.calculate_var_single_stock(stock, confidence, horizon)
                    current_price = self.prices[stock].iloc[-1]
                    var_dollar = abs(var_scaled * current_price)
                    print(f"  {horizon}-day VaR ({confidence*100:.0f}% confidence): {var_scaled:.4f} (${var_dollar:.2f})")
            worst_cases = self.get_worst_case_scenarios(stock, 3)
            print(f"  Worst 3 Historical Days:")
            for date, return_val in worst_cases.items():
                print(f"    {date.strftime('%Y-%m-%d')}: {return_val:.4f} ({return_val*100:.2f}%)")
        print(f"\n2. PORTFOLIO VaR ANALYSIS")
        print("-" * 50)
        n_stocks = len(self.prices.columns)
        equal_weights = np.array([1/n_stocks] * n_stocks)
        print(f"Equal-Weighted Portfolio ({', '.join(self.prices.columns)}):")
        print(f"Weights: {dict(zip(self.prices.columns, equal_weights))}")
        for confidence in confidence_levels:
            for horizon in time_horizons:
                var_1day, var_scaled, portfolio_returns = self.calculate_portfolio_var(
                    equal_weights, confidence, horizon
                )
                portfolio_value = 10000
                var_dollar = abs(var_scaled * portfolio_value)
                print(f"  {horizon}-day Portfolio VaR ({confidence*100:.0f}% confidence): {var_scaled:.4f} (${var_dollar:.2f})")
        print(f"\n3. DIVERSIFICATION ANALYSIS")
        print("-" * 50)
        individual_vars_sum = 0
        for stock in self.prices.columns:
            var_1day, var_scaled = self.calculate_var_single_stock(stock, 0.95, 1)
            individual_vars_sum += abs(var_scaled) * equal_weights[list(self.prices.columns).index(stock)]
        portfolio_var_1day, _, _ = self.calculate_portfolio_var(equal_weights, 0.95, 1)
        diversification_benefit = individual_vars_sum - abs(portfolio_var_1day)
        print(f"Sum of Individual VaRs: {individual_vars_sum:.4f}")
        print(f"Portfolio VaR: {abs(portfolio_var_1day):.4f}")
        print(f"Diversification Benefit: {diversification_benefit:.4f} ({diversification_benefit/individual_vars_sum*100:.1f}% reduction)")
    
    def plot_var_analysis(self):
        stock = self.prices.columns[0]

        # --- FIGURE 1: Single Stock Analysis (Histogram + Price Evolution) ---
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram with VaR lines
        axes1[0].hist(self.returns[stock], bins=50, alpha=0.7, color='blue')
        var_95_1day, _ = self.calculate_var_single_stock(stock, 0.95, 1)
        var_99_1day, _ = self.calculate_var_single_stock(stock, 0.99, 1)
        axes1[0].axvline(var_95_1day, color='red', linestyle='--', label=f'95% VaR: {var_95_1day:.4f}')
        axes1[0].axvline(var_99_1day, color='darkred', linestyle='--', label=f'99% VaR: {var_99_1day:.4f}')
        axes1[0].set_title(f'{stock} - Return Distribution with VaR')
        axes1[0].set_xlabel('Daily Returns')
        axes1[0].set_ylabel('Frequency')
        axes1[0].legend()

        # Price evolution
        axes1[1].plot(self.prices.index, self.prices[stock], color='green')
        axes1[1].set_title(f'{stock} - Price Evolution')
        axes1[1].set_xlabel('Date')
        axes1[1].set_ylabel('Price ($)')
        axes1[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # --- FIGURE 2: Portfolio Analysis (Portfolio Histogram + VaR Horizon) ---
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        equal_weights = np.array([0.25] * len(self.prices.columns))
        _, _, portfolio_returns = self.calculate_portfolio_var(equal_weights, 0.95, 1)

        # Portfolio histogram
        axes2[0].hist(portfolio_returns, bins=50, alpha=0.7, color='purple')
        portfolio_var_95, _, _ = self.calculate_portfolio_var(equal_weights, 0.95, 1)
        axes2[0].axvline(portfolio_var_95, color='red', linestyle='--', label=f'Portfolio 95% VaR: {portfolio_var_95:.4f}')
        axes2[0].set_title('Portfolio - Return Distribution with VaR')
        axes2[0].set_xlabel('Portfolio Returns')
        axes2[0].set_ylabel('Frequency')
        axes2[0].legend()

        # VaR vs Time Horizon
        horizons = [1, 2, 3, 4, 5, 10, 15, 20]
        var_values_95, var_values_99 = [], []
        for horizon in horizons:
            _, var_95_scaled = self.calculate_var_single_stock(stock, 0.95, horizon)
            _, var_99_scaled = self.calculate_var_single_stock(stock, 0.99, horizon)
            var_values_95.append(abs(var_95_scaled))
            var_values_99.append(abs(var_99_scaled))

        axes2[1].plot(horizons, var_values_95, marker='o', label='95% VaR', color='blue')
        axes2[1].plot(horizons, var_values_99, marker='s', label='99% VaR', color='red')
        axes2[1].set_title(f'{stock} - VaR vs Time Horizon')
        axes2[1].set_xlabel('Time Horizon (days)')
        axes2[1].set_ylabel('VaR (absolute)')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Generating sample stock market data...")
    stock_prices = generate_sample_data()
    print(f"Generated data for {len(stock_prices.columns)} stocks over {len(stock_prices)} trading days")
    print(f"Date range: {stock_prices.index[0].strftime('%Y-%m-%d')} to {stock_prices.index[-1].strftime('%Y-%m-%d')}")
    print()
    var_calculator = VaRCalculator(stock_prices)
    var_calculator.create_var_report()
    print("\nGenerating VaR visualizations...")
    var_calculator.plot_var_analysis()
    print("\n" + "="*80)
    print("VaR ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKEY INSIGHTS FOR QUANTITATIVE FINANCE:")
    print("1. VaR measures potential loss with a given confidence level")
    print("2. Historical method uses past data to predict future risk")
    print("3. Diversification reduces portfolio risk compared to individual stocks")
    print("4. Higher confidence levels result in higher VaR estimates")
    print("5. VaR scales with square root of time (for longer horizons)")
