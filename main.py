"""
# Адаптивна торгова стратегія з використанням Backtrader
# Автор: Клод 3.7 Sonnet
# Дата: 08.05.2025

# Стратегія використовує комбінацію технічних індикаторів для торгівлі акціями Apple:
# 1. MACD (Moving Average Convergence Divergence)
# 2. RSI (Relative Strength Index)
# 3. Волатильність (ATR - Average True Range)
# 4. Bollinger Bands
"""

import backtrader as bt
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Встановлюємо параметри графіків
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True

# -----------------------------------------------------------------------------
# 1. DATA ACCESS/CLEANING
# -----------------------------------------------------------------------------

# Функція для завантаження та попередньої обробки даних
def get_data(ticker='AAPL', start_date='2015-01-01', end_date='2025-01-01'):
    """
    Завантаження історичних даних акцій з Yahoo Finance та їх попередня обробка
    
    Параметри:
    ----------
    ticker : str
        Символ акції для завантаження
    start_date : str
        Початкова дата у форматі 'YYYY-MM-DD'
    end_date : str
        Кінцева дата у форматі 'YYYY-MM-DD'
        
    Повертає:
    ---------
    df : pd.DataFrame
        Оброблений датафрейм з даними акцій
    """
    print(f"Завантаження даних для {ticker} з {start_date} по {end_date}...")
    
    # Завантаження даних з Yahoo Finance
    from curl_cffi import requests
    session = requests.Session(impersonate="chrome")

    df = yf.download(ticker, start=start_date, end=end_date, session=session)
    
    # Перевірка на пропущені значення
    if df.isnull().sum().sum() > 0:
        print(f"Знайдено {df.isnull().sum().sum()} пропущених значень. Виконуємо заповнення...")
        df.fillna(method='ffill', inplace=True)  # Заповнення пропущених значень
    
    # Обчислення додаткових стовпців для аналізу
    df['Returns'] = df['Close'].pct_change()  # Денна дохідність
    df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))  # Логарифмічна дохідність
    
    # Обчислення ковзних середніх для різних періодів
    windows = [5, 10, 20, 50, 200]
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Обчислення волатильності (стандартне відхилення за 20 днів)
    df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
    
    # Обчислення Z-показника для ціни закриття за останні 20 днів
    df['Close_Z_20d'] = df['Close'].rolling(window=20).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    # Видалення рядків з пропущеними значеннями після обчислень
    df.dropna(inplace=True)
    
    print(f"Дані успішно завантажені та оброблені. Кількість записів: {len(df)}")
    return df

# Завантаження даних для Apple
data = get_data(ticker='AAPL', start_date='2015-01-01', end_date='2022-01-01')

# Збереження даних для подальшого використання
data.to_csv('apple_data.csv')

# Візуалізація часових рядів
plt.figure(figsize=(14, 10))

# Графік 1: Ціна закриття та ковзні середні
plt.subplot(3, 1, 1)
plt.plot(data.index, data['Close'], label='Ціна закриття', linewidth=1.5)
plt.plot(data.index, data['SMA_20'], label='SMA 20', linewidth=1.5)
plt.plot(data.index, data['SMA_50'], label='SMA 50', linewidth=1.5)
plt.plot(data.index, data['SMA_200'], label='SMA 200', linewidth=1.5)
plt.title('AAPL: Ціна закриття та ковзні середні')
plt.legend()
plt.grid(True)

# Графік 2: Денна дохідність
plt.subplot(3, 1, 2)
plt.plot(data.index, data['Returns'], label='Денна дохідність', color='green', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('AAPL: Денна дохідність')
plt.legend()
plt.grid(True)

# Графік 3: Волатильність (20-денне вікно)
plt.subplot(3, 1, 3)
plt.plot(data.index, data['Volatility_20d'], label='20-денна волатильність', color='orange')
plt.title('AAPL: 20-денна волатильність')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('apple_data_visualization.png')
plt.close()

# Додатковий аналіз даних
print("Статистичні характеристики даних:")
stats = data[['Close', 'Volume', 'Returns', 'Volatility_20d']].describe()
print(stats)

# Дослідження кореляцій
correlation_matrix = data[['Close', 'Volume', 'Returns', 'Volatility_20d', 'SMA_20', 'SMA_50']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Кореляційна матриця')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# -----------------------------------------------------------------------------
# 2. PRE-TRADE ANALYSIS
# -----------------------------------------------------------------------------

# Аналіз сезонності за днями тижня
data['DayOfWeek'] = data.index.dayofweek
day_of_week_returns = data.groupby('DayOfWeek')['Returns'].mean() * 100  # переведення у відсотки

plt.figure(figsize=(10, 6))
day_of_week_returns.plot(kind='bar', color='skyblue')
plt.title('Середня дохідність за днями тижня (%)')
plt.xlabel('День тижня (0=Понеділок, 4=П\'ятниця)')
plt.ylabel('Середня дохідність (%)')
plt.grid(True, axis='y')
plt.savefig('day_of_week_returns.png')
plt.close()

# Аналіз сезонності за місяцями
data['Month'] = data.index.month
monthly_returns = data.groupby('Month')['Returns'].mean() * 100  # переведення у відсотки

plt.figure(figsize=(10, 6))
monthly_returns.plot(kind='bar', color='lightgreen')
plt.title('Середня дохідність за місяцями (%)')
plt.xlabel('Місяць (1=Січень, 12=Грудень)')
plt.ylabel('Середня дохідність (%)')
plt.grid(True, axis='y')
plt.savefig('monthly_returns.png')
plt.close()

# Аналіз волатильності за місяцями
monthly_volatility = data.groupby('Month')['Volatility_20d'].mean() * 100  # переведення у відсотки

plt.figure(figsize=(10, 6))
monthly_volatility.plot(kind='bar', color='salmon')
plt.title('Середня волатильність за місяцями (%)')
plt.xlabel('Місяць (1=Січень, 12=Грудень)')
plt.ylabel('Середня волатильність (%)')
plt.grid(True, axis='y')
plt.savefig('monthly_volatility.png')
plt.close()

# Аналіз автокореляції доходності
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(10, 6))
autocorrelation_plot(data['Returns'].dropna())
plt.title('Автокореляція денної дохідності')
plt.grid(True)
plt.savefig('autocorrelation.png')
plt.close()

# Аналіз розподілу доходності
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['Returns'].dropna(), kde=True, bins=50)
plt.title('Розподіл денної доходності')
plt.xlabel('Доходність')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
import scipy.stats as stats
stats.probplot(data['Returns'].dropna(), plot=plt)
plt.title('Q-Q графік доходності')

plt.tight_layout()
plt.savefig('returns_distribution.png')
plt.close()

# Визначення статистики розподілу доходності
from scipy.stats import skew, kurtosis

returns_skew = skew(data['Returns'].dropna())
returns_kurtosis = kurtosis(data['Returns'].dropna())

print(f"Асиметрія розподілу доходності: {returns_skew:.4f}")
print(f"Ексцес розподілу доходності: {returns_kurtosis:.4f}")
print(f"Нормальний розподіл має асиметрію 0 і ексцес 0.")

# -----------------------------------------------------------------------------
# 3. TRADING SIGNAL
# -----------------------------------------------------------------------------

# Клас для створення індикатора сили тренду
class TrendStrengthIndicator(bt.Indicator):
    lines = ('trend_strength',)
    params = (('period', 20),)
    
    def __init__(self):
        self.ema_short = bt.indicators.EMA(period=self.p.period // 2)
        self.ema_long = bt.indicators.EMA(period=self.p.period)
        self.rsi = bt.indicators.RSI(period=self.p.period)
        self.atr = bt.indicators.ATR(period=self.p.period)
        
    def next(self):
        # Обчислення відносної різниці між коротким і довгим EMA
        ema_diff = (self.ema_short[0] - self.ema_long[0]) / self.ema_long[0]
        
        # Нормалізація RSI для отримання значення від -1 до 1
        rsi_norm = (self.rsi[0] - 50) / 50
        
        # Комбінація сигналів для оцінки сили тренду
        self.lines.trend_strength[0] = ema_diff * rsi_norm

# Клас розширеного індикатора сили тренду та волатильності
class EnhancedTrendIndicator(bt.Indicator):
    """Розширений індикатор для визначення сили тренду, ринкових умов та волатильності"""
    lines = ('trend_power', 'volatility_ratio', 'market_regime', 'momentum')
    params = (
        ('ema_short', 10),
        ('ema_medium', 30),
        ('ema_long', 50),
        ('atr_period', 14),
        ('vol_period', 20),
        ('rsi_period', 14),
        ('adx_period', 14),
    )
    
    def __init__(self):
        # Основні ЕМА для визначення тренду
        self.ema_short = bt.indicators.EMA(period=self.p.ema_short)
        self.ema_medium = bt.indicators.EMA(period=self.p.ema_medium)
        self.ema_long = bt.indicators.EMA(period=self.p.ema_long)
        
        # Індикатори тренду
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.adx = bt.indicators.DirectionalMovement(period=self.p.adx_period)
        
        # Волатильність
        self.stddev = bt.indicators.StdDev(period=self.p.vol_period)
        self.avg_price = bt.indicators.MovAv.SMA(self.data.close, period=self.p.vol_period)
        
        # Моментум та імпульс
        self.mom = bt.indicators.Momentum(period=12)
        self.macd = bt.indicators.MACD()
        
    def next(self):
        # Визначення сили тренду на основі комбінації індикаторів
        ema_alignment = 0
        if self.ema_short[0] > self.ema_medium[0] > self.ema_long[0]:
            ema_alignment = 1  # Висхідний тренд
        elif self.ema_short[0] < self.ema_medium[0] < self.ema_long[0]:
            ema_alignment = -1  # Низхідний тренд
            
        # Нормалізований RSI (-1 до 1)
        rsi_norm = (self.rsi[0] - 50) / 50
        
        # ADX для визначення сили тренду (0-100, вищі значення - сильніший тренд)
        adx_strength = self.adx.adx[0] / 100
        
        # Комбінований показник сили тренду
        self.lines.trend_power[0] = ema_alignment * adx_strength * (1 + abs(rsi_norm))
        
        # Відносна волатильність (порівняння з попередніми періодами)
        if len(self) > self.p.vol_period:
            avg_volatility = self.stddev[-self.p.vol_period:].mean()
            if avg_volatility > 0:
                self.lines.volatility_ratio[0] = self.stddev[0] / avg_volatility
            else:
                self.lines.volatility_ratio[0] = 1.0
        else:
            self.lines.volatility_ratio[0] = 1.0
            
        # Ринковий режим (-1: сильно ведмежий, 0: нейтральний, 1: сильно бичачий)
        self.lines.market_regime[0] = ema_alignment * (adx_strength * 0.5 + (rsi_norm * 0.5))
        
        # Показник моментуму
        self.lines.momentum[0] = (self.mom[0] / self.data.close[0]) * (self.macd.macd[0] / self.data.close[0] * 100)


# Клас вдосконаленої волатильності
class VolatilityAdapter(bt.Indicator):
    """Індикатор для динамічної адаптації параметрів стратегії залежно від ринкової волатильності"""
    lines = ('multiplier', 'atr_ratio', 'market_state')
    params = (
        ('atr_period', 14),
        ('look_back', 100),  # Для порівняння поточної волатильності з історичною
        ('thresh_low', 0.5),  # Поріг низької волатильності
        ('thresh_high', 1.5),  # Поріг високої волатильності
    )
    
    def __init__(self):
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.atr_sma = bt.indicators.SMA(self.atr, period=self.p.look_back)
        
    def next(self):
        # Обчислення відношення поточного ATR до середнього
        if self.atr_sma[0] > 0:
            self.lines.atr_ratio[0] = self.atr[0] / self.atr_sma[0]
        else:
            self.lines.atr_ratio[0] = 1.0
            
        # Визначення стану ринку на основі волатильності
        if self.lines.atr_ratio[0] < self.p.thresh_low:
            # Низька волатильність - контрактація ринку, можливий прорив
            self.lines.market_state[0] = -1
            self.lines.multiplier[0] = 0.75  # Зменшуємо ризик в очікуванні прориву
        elif self.lines.atr_ratio[0] > self.p.thresh_high:
            # Висока волатильність - можливі помилкові сигнали, але також можливості
            self.lines.market_state[0] = 1
            self.lines.multiplier[0] = 1.25  # Збільшуємо поріг для входу, але збільшуємо розмір позиції
        else:
            # Нормальна волатильність
            self.lines.market_state[0] = 0
            self.lines.multiplier[0] = 1.0


# Розширена стратегія з більшою кількістю торгових сигналів
class EnhancedAdaptiveStrategy(bt.Strategy):
    """Розширена адаптивна торгова стратегія з додатковими індикаторами і більшою кількістю сигналів"""
    
    params = (
        # Базові параметри
        ('risk_per_trade', 0.02),  # 2% ризику на угоду
        ('stop_loss_atr', 2.0),    # Стоп-лосс на рівні 2 ATR
        ('take_profit_atr', 3.0),  # Тейк-профіт на рівні 3 ATR
        
        # Налаштування індикаторів
        ('atr_period', 14),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # Нові параметри
        ('adx_period', 14),
        ('adx_threshold', 25),     # Мінімальний рівень ADX для підтвердження тренду
        ('cci_period', 20),
        ('cci_entry', 100),        # Рівень CCI для входу
        ('stoch_period', 14),
        ('stoch_slowk', 3),
        ('stoch_slowd', 3),
        ('ichimoku_convkijun', 9), # Параметр Ichimoku Conversion Line
        ('ichimoku_kijun', 26),    # Параметр Ichimoku Base Line
        
        # Параметри для обробки ринкових станів
        ('volatility_look_back', 100),
        ('trend_detection_period', 50),
        
        # Додаткові умови входу
        ('allow_countertrend', True),  # Дозволяти контртрендові операції в певних умовах
        ('signal_threshold', 2),       # Мінімальна кількість підтверджуючих сигналів для входу
        ('enable_filters', True),      # Включення додаткових фільтрів для зменшення шуму
        
        # Параметри для різних типів сигналів
        ('enable_breakout', True),     # Сигнали прориву
        ('enable_pullback', True),     # Сигнали відкату
        ('enable_reversal', True),     # Сигнали розвороту
        ('enable_volatility', True),   # Сигнали, засновані на волатильності
        
        # Параметри для часових рамок входу
        ('time_filter', False),        # Фільтр за часом доби
        ('day_filter', False),         # Фільтр за днем тижня
    )
    
    def log(self, txt, dt=None):
        """Функція для логування"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    
    def __init__(self):
        """Ініціалізація стратегії"""
        # Зберігаємо посилання на дані
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataopen = self.datas[0].open
        self.datavolume = self.datas[0].volume
        
        # Відстежуємо замовлення і позиції
        self.order = None
        self.trades_count = 0
        self.profitable_trades = 0
        self.stop_loss_triggered = 0
        self.take_profit_triggered = 0
        self.order_type = None  # 'buy', 'sell' або None
        
        # Рівні стоп-лосу та тейк-профіту
        self.stop_loss = None
        self.take_profit = None
        
        # Зберігаємо сигнали для виведення
        self.buy_signals = []
        self.sell_signals = []
        
        # Основні індикатори тренду
        self.sma20 = bt.indicators.SMA(period=20)
        self.sma50 = bt.indicators.SMA(period=50)
        self.sma200 = bt.indicators.SMA(period=200)
        self.ema9 = bt.indicators.EMA(period=9)
        self.ema21 = bt.indicators.EMA(period=21)
        
        # Індикатори імпульсу
        self.macd = bt.indicators.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # Осцилятори
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.stochastic = bt.indicators.Stochastic(
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_slowk,
            period_dslow=self.p.stoch_slowd
        )
        self.cci = bt.indicators.CCI(period=self.p.cci_period)
        
        # Межі волатильності
        self.bollinger = bt.indicators.BollingerBands(
            period=self.p.bb_period,
            devfactor=self.p.bb_dev
        )
        self.keltner = bt.indicators.KeltnerChannel(period=20, multiplier=2.0)
        
        # Індикатори сили тренду
        self.adx = bt.indicators.DirectionalMovement(period=self.p.adx_period)
        self.aroon = bt.indicators.AroonIndicator(period=25)
        
        # Ichimoku Cloud для ідентифікації рівнів підтримки/опору та тренду
        self.ichimoku = bt.indicators.Ichimoku(
            tenkan=self.p.ichimoku_convkijun,
            kijun=self.p.ichimoku_kijun
        )
        
        # ATR для визначення волатильності і розміру позиції
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        
        # Власні індикатори для відстеження ринкових умов
        self.trend_indicator = EnhancedTrendIndicator()
        self.volatility_adapter = VolatilityAdapter(
            atr_period=self.p.atr_period, 
            look_back=self.p.volatility_look_back
        )
        
        # Додаткові індикатори
        self.heikin_ashi = bt.indicators.HeikinAshi()
        self.psar = bt.indicators.ParabolicSAR()
        self.vortex = bt.indicators.Vortex(period=14)  # Для виявлення початку нових трендів
        
        # Індикатор об'єму для підтвердження трендів
        self.obv = bt.indicators.OnBalanceVolume()  # On Balance Volume
        self.cmf = bt.indicators.ChaikinMoneyFlow()  # Chaikin Money Flow
        
        # Історія ATR для обчислення динамічної волатильності
        self.atr_history = []
        
    def notify_order(self, order):
        """Обробка повідомлень про замовлення"""
        if order.status in [order.Submitted, order.Accepted]:
            # Замовлення подано/прийнято - нічого не робимо
            return
            
        # Перевірка, чи виконано або скасовано замовлення
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Ціна: {order.executed.price:.2f}, Комісія: {order.executed.comm:.2f}, Тип: {self.order_type}')
                # Встановлення стоп-лосу та тейк-профіту
                atr_value = self.atr[0]
                vol_adj = self.volatility_adapter.multiplier[0]  # Адаптація до волатильності
                
                self.stop_loss = order.executed.price * (1 - self.p.stop_loss_atr * atr_value * vol_adj / order.executed.price)
                self.take_profit = order.executed.price * (1 + self.p.take_profit_atr * atr_value * vol_adj / order.executed.price)
                
                self.log(f'Встановлено SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}')
                
            else:  # Sell
                self.log(f'SELL EXECUTED, Ціна: {order.executed.price:.2f}, Комісія: {order.executed.comm:.2f}, Тип: {self.order_type}')
                # Для шортів - стоп вище, тейк нижче
                atr_value = self.atr[0]
                vol_adj = self.volatility_adapter.multiplier[0]
                
                self.stop_loss = order.executed.price * (1 + self.p.stop_loss_atr * atr_value * vol_adj / order.executed.price)
                self.take_profit = order.executed.price * (1 - self.p.take_profit_atr * atr_value * vol_adj / order.executed.price)
                
                self.log(f'Встановлено SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}')
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Замовлення скасовано/відхилено')
            
        # Скидаємо замовлення
        self.order = None
    
    def notify_trade(self, trade):
        """Обробка повідомлень про торгівлю"""
        if not trade.isclosed:
            return
            
        self.trades_count += 1
        if trade.pnl > 0:
            self.profitable_trades += 1
            
        self.log(f'ОПЕРАЦІЯ ЗАКРИТА, ПРИБУТОК/ЗБИТОК: {trade.pnl:.2f}, ЧИСТИЙ: {trade.pnlcomm:.2f}')
        
    def get_position_size(self):
        """Визначення розміру позиції на основі ризику та волатильності"""
        risk_amount = self.broker.getvalue() * self.p.risk_per_trade
        atr_value = self.atr[0]
        
        # Адаптація розміру позиції залежно від волатильності та ринкового стану
        vol_adj = self.volatility_adapter.multiplier[0]
        trend_strength = abs(self.trend_indicator.trend_power[0])
        
        # Збільшуємо розмір у напрямку сильного тренду, зменшуємо для слабкого
        if trend_strength > 0.7:  # Сильний тренд
            trend_adj = 1.2
        elif trend_strength < 0.3:  # Слабкий тренд
            trend_adj = 0.8
        else:
            trend_adj = 1.0
            
        # Розраховуємо розмір з урахуванням усіх факторів
        if atr_value > 0:
            # Формула: Ризик / (Стоп-лосс у пунктах * коефіцієнт волатильності * коефіцієнт тренду)
            shares = risk_amount / (atr_value * self.p.stop_loss_atr * vol_adj * trend_adj)
            return max(1, int(shares))  # Мінімум 1 акція
        else:
            return int(risk_amount / self.dataclose[0] * 0.01)  # Запасний варіант
            
    def should_buy(self):
        """Розширений аналіз умов для відкриття довгої позиції"""
        signals = []  # Список сигналів
        
        # === СИГНАЛИ ТРЕНДУ ===
        
        # Перевірка висхідного тренду за різними часовими рамками
        trend_sma = self.sma20[0] > self.sma50[0] and self.sma50[0] > self.sma200[0]
        trend_ema = self.ema9[0] > self.ema21[0]
        
        # Сила тренду від власного індикатора
        trend_strength = self.trend_indicator.trend_power[0]
        strong_uptrend = trend_strength > 0.5
        
        # Перевірка напрямку тренду за ADX і DI
        adx_trend = self.adx.adx[0] > self.p.adx_threshold and self.adx.plus_di[0] > self.adx.minus_di[0]
        
        # === СИГНАЛИ ІМПУЛЬСУ ===
        
        # MACD перетин
        macd_cross_up = self.macd.macd[-1] < self.macd.signal[-1] and self.macd.macd[0] > self.macd.signal[0]
        
        # MACD вище нуля (позитивний моментум)
        macd_above_zero = self.macd.macd[0] > 0
        
        # === ОСЦИЛЯТОРИ ТА СИГНАЛИ ПЕРЕКУПЛЕНОСТІ/ПЕРЕПРОДАНОСТІ ===
        
        # RSI виходить із зони перепроданості
        rsi_bottom = self.rsi[-1] < self.p.rsi_oversold and self.rsi[0] > self.p.rsi_oversold
        
        # RSI росте
        rsi_rising = self.rsi[0] > self.rsi[-1] > self.rsi[-2]
        
        # Стохастик виходить із зони перепроданості
        stoch_bottom = self.stochastic.percK[-1] < 20 and self.stochastic.percK[0] > 20
        
        # CCI виходить із зони перепроданості
        cci_bottom = self.cci[-1] < -self.p.cci_entry and self.cci[0] > -self.p.cci_entry
        
        # === СИГНАЛИ ПРОРИВУ ===
        
        # Прорив верхньої межі полоси Боллінджера після консолідації
        bb_breakout = self.dataclose[-1] < self.bollinger.lines.top[-1] and self.dataclose[0] > self.bollinger.lines.top[0]
        
        # Відскок від нижньої смуги Боллінджера
        bb_bounce = self.dataclose[-1] < self.bollinger.lines.bot[-1] and self.dataclose[0] > self.bollinger.lines.bot[0]
        
        # Прорив каналу Кельтнера
        keltner_breakout = self.dataclose[-1] < self.keltner.top[-1] and self.dataclose[0] > self.keltner.top[0]
        
        # === СИГНАЛИ СВІЧКОВИХ ПАТЕРНІВ (спрощено) ===
        
        # Молот (спрощено)
        hammer = (self.dataclose[0] > self.dataopen[0] and 
                 (self.dataclose[0] - self.datalow[0]) > 2 * (self.dataclose[0] - self.dataopen[0]) and
                 (self.datahigh[0] - self.dataclose[0]) < 0.5 * (self.dataclose[0] - self.dataopen[0]))
        
        # Поглинання (спрощено)
        bullish_engulfing = (self.dataopen[-1] > self.dataclose[-1] and  # Попередня свічка червона
                            self.dataopen[0] < self.dataclose[0] and     # Поточна свічка зелена
                            self.dataopen[0] <= self.dataclose[-1] and   # Відкриття нижче закриття попередньої
                            self.dataclose[0] >= self.dataopen[-1])      # Закриття вище відкриття попередньої
        
        # === СИГНАЛИ ОБ'ЄМУ ===
        
        # Зростання OBV
        obv_rising = self.obv[0] > self.obv[-1] > self.obv[-2]
        
        # Позитивний Chaikin Money Flow
        cmf_positive = self.cmf[0] > 0
        
        # === СИГНАЛИ ICHIMOKU ===
        
        # Ціна вище хмари
        ichimoku_above_cloud = (self.dataclose[0] > self.ichimoku.lines.senkou_span_a[0] and 
                              self.dataclose[0] > self.ichimoku.lines.senkou_span_b[0])
        
        # Перетин ліній Conversion і Base
        ichimoku_cross = (self.ichimoku.lines.tenkan_sen[-1] < self.ichimoku.lines.kijun_sen[-1] and 
                       self.ichimoku.lines.tenkan_sen[0] > self.ichimoku.lines.kijun_sen[0])
        
        # === СИГНАЛИ ПАРАБОЛІКА І VORTEX ===
        
        # Зміна напряму Parabolic SAR
        psar_flip = self.psar[-1] > self.dataclose[-1] and self.psar[0] < self.dataclose[0]
        
        # Перетин ліній Vortex
        vortex_cross = self.vortex.lines.vi_plus[-1] < self.vortex.lines.vi_minus[-1] and self.vortex.lines.vi_plus[0] > self.vortex.lines.vi_minus[0]
        
        # === ЗБІР ТА АНАЛІЗ СИГНАЛІВ ===
        
        # Додаємо всі сигнали в список
        signals.extend([
            trend_sma, trend_ema, strong_uptrend, adx_trend,
            macd_cross_up, macd_above_zero,
            rsi_bottom, rsi_rising, stoch_bottom, cci_bottom,
            bb_breakout, bb_bounce, keltner_breakout,
            hammer, bullish_engulfing,
            obv_rising, cmf_positive,
            ichimoku_above_cloud, ichimoku_cross,
            psar_flip, vortex_cross
        ])
        
        # Рахуємо загальну кількість позитивних сигналів
        signal_count = sum(1 for signal in signals if signal)
        
        # Визначаємо текстовий опис сигналу для логування
        signal_desc = ""
        for idx, signal_name in enumerate([
            "Тренд SMA", "Тренд EMA", "Сильний тренд", "ADX тренд",
            "MACD перетин", "MACD над 0",
            "RSI від низу", "RSI росте", "Стохастик від низу", "CCI від низу",
            "BB прорив", "BB відскок", "Keltner прорив",
            "Молот", "Поглинання",
            "OBV зростає", "CMF позитивний",
            "Над хмарою", "Ichimoku перетин",
            "PSAR розворот", "Vortex перетин"
        ]):
            if signals[idx]:
                signal_desc += f"{signal_name}, "
                
        # Додатково зважуємо сигнали залежно від ринкових умов
        market_state = self.volatility_adapter.market_state[0]
        
        # Зменшуємо поріг сигналів для входу при низькій волатильності
        if market_state == -1:  # Низька волатильність
            signal_threshold = self.p.signal_threshold - 1
        elif market_state == 1:  # Висока волатильність
            signal_threshold = self.p.signal_threshold + 1
        else:
            signal_threshold = self.p.signal_threshold
            
        # Перевірка наявності мінімальної кількості сигналів
        if signal_count >= signal_threshold:
            # Зберігаємо сигнал для візуалізації з його описом
            self.buy_signals.append((len(self) - 1, self.dataclose[0]))
            self.log(f"BUY SIGNAL: {signal_count} сигналів: {signal_desc}")
            self.order_type = "buy"
            return True
            
        return False
    
    def should_sell(self):
        """Розширений аналіз умов для відкриття короткої позиції"""
        signals = []  # Список сигналів
        
        # === СИГНАЛИ ТРЕНДУ ===
        
        # Перевірка низхідного тренду за різними часовими рамками
        trend_sma = self.sma20[0] < self.sma50[0] and self.sma50[0] < self.sma200[0]
        trend_ema = self.ema9[0] < self.ema21[0]
        
        # Сила тренду від власного індикатора
        trend_strength = self.trend_indicator.trend_power[0]
        strong_downtrend = trend_strength < -0.5
        
        # Перевірка напрямку тренду за ADX і DI
        adx_trend = self.adx.adx[0] > self.p.adx_threshold and self.adx.minus_di[0] > self.adx.plus_di[0]
        
        # === СИГНАЛИ ІМПУЛЬСУ ===
        
        # MACD перетин
        macd_cross_down = self.macd.macd[-1] > self.macd.signal[-1] and self.macd.macd[0] < self.macd.signal[0]
        
        # MACD нижче нуля (негативний моментум)
        macd_below_zero = self.macd.macd[0] < 0
        
        # === ОСЦИЛЯТОРИ ТА СИГНАЛИ ПЕРЕКУПЛЕНОСТІ/ПЕРЕПРОДАНОСТІ ===
        
        # RSI виходить із зони перекупленості
        rsi_top = self.rsi[-1] > self.p.rsi_overbought and self.rsi[0] < self.p.rsi_overbought
        
        # RSI падає
        rsi_falling = self.rsi[0] < self.rsi[-1] < self.rsi[-2]
        
        # Стохастик виходить із зони перекупленості
        stoch_top = self.stochastic.percK[-1] > 80 and self.stochastic.percK[0] < 80
        
        # CCI виходить із зони перекупленості
        cci_top = self.cci[-1] > self.p.cci_entry and self.cci[0] < self.p.cci_entry
        
        # === СИГНАЛИ ПРОРИВУ ===
        
        # Прорив нижньої межі полоси Боллінджера після консолідації
        bb_breakout = self.dataclose[-1] > self.bollinger.lines.bot[-1] and self.dataclose[0] < self.bollinger.lines.bot[0]
        
        # Відскок від верхньої смуги Боллінджера
        bb_bounce = self.dataclose[-1] > self.bollinger.lines.top[-1] and self.dataclose[0] < self.bollinger.lines.top[0]
        
        # Прорив каналу Кельтнера (вниз)
        keltner_breakout = self.dataclose[-1] > self.keltner.bot[-1] and self.dataclose[0] < self.keltner.bot[0]
        
        # === СИГНАЛИ СВІЧКОВИХ ПАТЕРНІВ (спрощено) ===
        
        # Повішений (спрощено)
        hanging_man = (self.dataclose[0] < self.dataopen[0] and 
                    (self.dataopen[0] - self.datalow[0]) > 2 * (self.dataopen[0] - self.dataclose[0]) and
                    (self.datahigh[0] - self.dataopen[0]) < 0.5 * (self.dataopen[0] - self.dataclose[0]))
        
        # Поглинання ведмеже (спрощено)
        bearish_engulfing = (self.dataopen[-1] < self.dataclose[-1] and  # Попередня свічка зелена
                          self.dataopen[0] > self.dataclose[0] and     # Поточна свічка червона
                          self.dataopen[0] >= self.dataclose[-1] and   # Відкриття вище закриття попередньої
                          self.dataclose[0] <= self.dataopen[-1])      # Закриття нижче відкриття попередньої
        
        # === СИГНАЛИ ОБ'ЄМУ ===
        
        # Падіння OBV
        obv_falling = self.obv[0] < self.obv[-1] < self.obv[-2]
        
        # Негативний Chaikin Money Flow
        cmf_negative = self.cmf[0] < 0
        
        # === СИГНАЛИ ICHIMOKU ===
        
        # Ціна нижче хмари
        ichimoku_below_cloud = (self.dataclose[0] < self.ichimoku.lines.senkou_span_a[0] and 
                             self.dataclose[0] < self.ichimoku.lines.senkou_span_b[0])
        
        # Перетин ліній Conversion і Base (вниз)
        ichimoku_cross = (self.ichimoku.lines.tenkan_sen[-1] > self.ichimoku.lines.kijun_sen[-1] and 
                       self.ichimoku.lines.tenkan_sen[0] < self.ichimoku.lines.kijun_sen[0])
        
        # === СИГНАЛИ ПАРАБОЛІКА І VORTEX ===
        
        # Зміна напряму Parabolic SAR (вгору)
        psar_flip = self.psar[-1] < self.dataclose[-1] and self.psar[0] > self.dataclose[0]
        
        # Перетин ліній Vortex (вниз)
        vortex_cross = self.vortex.lines.vi_plus[-1] > self.vortex.lines.vi_minus[-1] and self.vortex.lines.vi_plus[0] < self.vortex.lines.vi_minus[0]
        
        # === ЗБІР ТА АНАЛІЗ СИГНАЛІВ ===
        
        # Додаємо всі сигнали в список
        signals.extend([
            trend_sma, trend_ema, strong_downtrend, adx_trend,
            macd_cross_down, macd_below_zero,
            rsi_top, rsi_falling, stoch_top, cci_top,
            bb_breakout, bb_bounce, keltner_breakout,
            hanging_man, bearish_engulfing,
            obv_falling, cmf_negative,
            ichimoku_below_cloud, ichimoku_cross,
            psar_flip, vortex_cross
        ])
        
        # Рахуємо загальну кількість позитивних сигналів
        signal_count = sum(1 for signal in signals if signal)
        
        # Визначаємо текстовий опис сигналу для логування
        signal_desc = ""
        for idx, signal_name in enumerate([
            "Тренд SMA", "Тренд EMA", "Сильний тренд", "ADX тренд",
            "MACD перетин", "MACD під 0",
            "RSI від верху", "RSI падає", "Стохастик від верху", "CCI від верху",
            "BB прорив", "BB відскок", "Keltner прорив",
            "Повішений", "Поглинання",
            "OBV падає", "CMF негативний",
            "Під хмарою", "Ichimoku перетин",
            "PSAR розворот", "Vortex перетин"
        ]):
            if signals[idx]:
                signal_desc += f"{signal_name}, "
                
        # Додатково зважуємо сигнали залежно від ринкових умов
        market_state = self.volatility_adapter.market_state[0]
        
        # Зменшуємо поріг сигналів для входу при низькій волатильності
        if market_state == -1:  # Низька волатильність
            signal_threshold = self.p.signal_threshold - 1
        elif market_state == 1:  # Висока волатильність
            signal_threshold = self.p.signal_threshold + 1
        else:
            signal_threshold = self.p.signal_threshold
            
        # Перевірка наявності мінімальної кількості сигналів
        if signal_count >= signal_threshold:
            # Зберігаємо сигнал для візуалізації з його описом
            self.sell_signals.append((len(self) - 1, self.dataclose[0]))
            self.log(f"SELL SIGNAL: {signal_count} сигналів: {signal_desc}")
            self.order_type = "sell"
            return True
            
        return False
        
    def check_exit_conditions(self):
        """Розширений аналіз умов для закриття позиції"""
        if not self.position:
            return False
            
        # Базові умови виходу по стоп-лосу та тейк-профіту
        if self.position.size > 0:  # Довга позиція
            # Перевірка стоп-лосу
            if self.stop_loss is not None and self.dataclose[0] < self.stop_loss:
                self.log(f'ЗАКРИТТЯ ЗА СТОП-ЛОСОМ: {self.dataclose[0]:.2f} < {self.stop_loss:.2f}')
                self.stop_loss_triggered += 1
                return True
                
            # Перевірка тейк-профіту
            if self.take_profit is not None and self.dataclose[0] > self.take_profit:
                self.log(f'ЗАКРИТТЯ ЗА ТЕЙК-ПРОФІТОМ: {self.dataclose[0]:.2f} > {self.take_profit:.2f}')
                self.take_profit_triggered += 1
                return True
                
            # === ДОДАТКОВІ УМОВИ ВИХОДУ ДЛЯ ДОВГОЇ ПОЗИЦІЇ ===
            
            # Трейлінг-стоп на основі ATR (якщо ціна падає на X% від максимуму, досягнутого після входу)
            if hasattr(self, 'bar_executed') and len(self) > self.bar_executed:
                # Знаходимо максимальну ціну з моменту входу
                max_price = max(self.dataclose.get(size=len(self) - self.bar_executed))
                trailing_stop = max_price * (1 - self.atr[0] * 1.5 / max_price)  # 1.5 ATR з максимуму
                
                if self.dataclose[0] < trailing_stop and max_price > self.position.price:
                    self.log(f'ЗАКРИТТЯ ЗА ТРЕЙЛІНГ-СТОПОМ: Максимум {max_price:.2f}, Поточна {self.dataclose[0]:.2f}, Стоп {trailing_stop:.2f}')
                    return True
            
            # Зміна тренду та сигнали розвороту
            trend_change = self.trend_indicator.trend_power[0] < -0.3 and self.trend_indicator.trend_power[-1] > 0
            
            # MACD перетин вниз
            macd_exit = self.macd.macd[-1] > self.macd.signal[-1] and self.macd.macd[0] < self.macd.signal[0]
            
            # RSI в зоні перекупленості і починає падати
            rsi_exit = self.rsi[0] > 70 and self.rsi[0] < self.rsi[-1]
            
            # Parabolic SAR перетин (зміна напрямку)
            psar_exit = self.psar[-1] < self.dataclose[-1] and self.psar[0] > self.dataclose[0]
            
            # Вихід на основі часу утримання позиції (якщо довго тримаємо без просування)
            time_exit = False
            if hasattr(self, 'bar_executed') and len(self) - self.bar_executed > 10:
                if self.dataclose[0] < self.position.price * 1.005:  # Менше 0.5% росту за 10 барів
                    time_exit = True
            
            # Комбінуємо умови виходу
            exit_signals = [trend_change, macd_exit, rsi_exit, psar_exit, time_exit]
            exit_count = sum(1 for signal in exit_signals if signal)
            
            if exit_count >= 2:  # Мінімум 2 сигнали для виходу
                self.log(f'ЗАКРИТТЯ ДОВГОЇ ПОЗИЦІЇ ЗА СИГНАЛАМИ: {"Тренд " if trend_change else ""}{"MACD " if macd_exit else ""}{"RSI " if rsi_exit else ""}{"PSAR " if psar_exit else ""}{"Час " if time_exit else ""}')
                return True
                
        else:  # Коротка позиція
            # Перевірка стоп-лосу
            if self.stop_loss is not None and self.dataclose[0] > self.stop_loss:
                self.log(f'ЗАКРИТТЯ ЗА СТОП-ЛОСОМ (SHORT): {self.dataclose[0]::.2f} > {self.stop_loss:.2f}')
                self.stop_loss_triggered += 1
                return True
                
            # Перевірка тейк-профіту
            if self.take_profit is not None and self.dataclose[0] < self.take_profit:
                self.log(f'ЗАКРИТТЯ ЗА ТЕЙК-ПРОФІТОМ (SHORT): {self.dataclose[0]:.2f} < {self.take_profit:.2f}')
                self.take_profit_triggered += 1
                return True
                
            # === ДОДАТКОВІ УМОВИ ВИХОДУ ДЛЯ КОРОТКОЇ ПОЗИЦІЇ ===
            
            # Трейлінг-стоп на основі ATR (якщо ціна росте на X% від мінімуму, досягнутого після входу)
            if hasattr(self, 'bar_executed') and len(self) > self.bar_executed:
                # Знаходимо мінімальну ціну з моменту входу
                min_price = min(self.dataclose.get(size=len(self) - self.bar_executed))
                trailing_stop = min_price * (1 + self.atr[0] * 1.5 / min_price)  # 1.5 ATR з мінімуму
                
                if self.dataclose[0] > trailing_stop and min_price < self.position.price:
                    self.log(f'ЗАКРИТТЯ ЗА ТРЕЙЛІНГ-СТОПОМ (SHORT): Мінімум {min_price:.2f}, Поточна {self.dataclose[0]:.2f}, Стоп {trailing_stop:.2f}')
                    return True
            
            # Зміна тренду та сигнали розвороту
            trend_change = self.trend_indicator.trend_power[0] > 0.3 and self.trend_indicator.trend_power[-1] < 0
            
            # MACD перетин вверх
            macd_exit = self.macd.macd[-1] < self.macd.signal[-1] and self.macd.macd[0] > self.macd.signal[0]
            
            # RSI в зоні перепроданості і починає рости
            rsi_exit = self.rsi[0] < 30 and self.rsi[0] > self.rsi[-1]
            
            # Parabolic SAR перетин (зміна напрямку)
            psar_exit = self.psar[-1] > self.dataclose[-1] and self.psar[0] < self.dataclose[0]
            
            # Вихід на основі часу утримання позиції (якщо довго тримаємо без просування)
            time_exit = False
            if hasattr(self, 'bar_executed') and len(self) - self.bar_executed > 10:
                if self.dataclose[0] > self.position.price * 0.995:  # Менше 0.5% падіння за 10 барів
                    time_exit = True
            
            # Комбінуємо умови виходу
            exit_signals = [trend_change, macd_exit, rsi_exit, psar_exit, time_exit]
            exit_count = sum(1 for signal in exit_signals if signal)
            
            if exit_count >= 2:  # Мінімум 2 сигнали для виходу
                self.log(f'ЗАКРИТТЯ КОРОТКОЇ ПОЗИЦІЇ ЗА СИГНАЛАМИ: {"Тренд " if trend_change else ""}{"MACD " if macd_exit else ""}{"RSI " if rsi_exit else ""}{"PSAR " if psar_exit else ""}{"Час " if time_exit else ""}')
                return True
                
        return False
        
    def next(self):
        """Основна логіка для кожного бару"""
        # Пропускаємо, якщо є активне замовлення
        if self.order:
            return
            
        # Зберігаємо поточне значення ATR у історію
        self.atr_history.append(self.atr[0])
        # Тримаємо історію не більше ніж останні 100 значень
        if len(self.atr_history) > 100:
            self.atr_history.pop(0)
            
        # Перевірка умов для закриття позиції
        if self.position and self.check_exit_conditions():
            self.log(f'ЗАКРИТТЯ ПОЗИЦІЇ, Ціна: {self.dataclose[0]:.2f}')
            self.order = self.close()
            return
            
        # Перевірка умов для відкриття позиції (якщо немає відкритих позицій)
        if not self.position:
            # Перевірка на покупку
            if self.should_buy():
                size = self.get_position_size()
                self.log(f'BUY ORDER, Ціна: {self.dataclose[0]:.2f}, Розмір: {size}')
                self.order = self.buy(size=size)
                
            # Перевірка на продаж (шорт)
            elif self.should_sell():
                size = self.get_position_size()
                self.log(f'SELL ORDER, Ціна: {self.dataclose[0]:.2f}, Розмір: {size}')
                self.order = self.sell(size=size)
                
    def stop(self):
        """Викликається в кінці бектесту"""
        self.log(f'Загальна кількість угод: {self.trades_count}')
        self.log(f'Прибуткових угод: {self.profitable_trades}')
        
        if self.trades_count > 0:
            win_rate = self.profitable_trades / self.trades_count * 100
            avg_profit = 0
            avg_loss = 0
            
            self.log(f'Вінрейт: {win_rate:.2f}%')
            
        self.log(f'Стоп-лосів спрацювало: {self.stop_loss_triggered}')
        self.log(f'Тейк-профітів спрацювало: {self.take_profit_triggered}')
        
        # Виведення додаткової статистики
        if hasattr(self, 'buy_signals') and hasattr(self, 'sell_signals'):
            self.log(f'Загальна кількість сигналів: Buy={len(self.buy_signals)}, Sell={len(self.sell_signals)}')

# Додатковий клас-спостерігач для збереження даних індикаторів
class StrategyDataObserver(bt.Observer):
    lines = ('macd', 'macd_signal', 'rsi', 'upper_band', 'lower_band')
    
    def next(self):
        self.lines.macd[0] = self._owner.macd.macd[0]
        self.lines.macd_signal[0] = self._owner.macd.signal[0]
        self.lines.rsi[0] = self._owner.rsi[0]
        self.lines.upper_band[0] = self._owner.bollinger.lines.top[0]
        self.lines.lower_band[0] = self._owner.bollinger.lines.bot[0]

# -----------------------------------------------------------------------------
# 4. TRADE EXECUTION
# -----------------------------------------------------------------------------

# Функція для запуску бектесту
def run_backtest(data_file, strategy_class, cash=100000.0, commission=0.001):
    """
    Виконання бектесту з вказаною стратегією
    
    Параметри:
    ----------
    data_file : str
        Шлях до файлу з даними
    strategy_class : bt.Strategy
        Клас стратегії
    cash : float
        Початковий капітал
    commission : float
        Комісія брокера
        
    Повертає:
    ---------
    cerebro : bt.Cerebro
        Екземпляр Cerebro з результатами бектесту
    """
    # Створення екземпляру Cerebro
    cerebro = bt.Cerebro()
    
    # Додавання стратегії
    cerebro.addstrategy(strategy_class)
    
    # Завантаження даних
    print(f"Завантаження даних з файлу {data_file}...")
    
    # Використання pandas для читання та підготовки даних для Backtrader
    import pandas as pd
    df = pd.read_csv(data_file, skiprows=[1, 2])  # Пропускаємо рядки з 'Ticker' та 'Date'
    
    # Перетворення стовпця з датами у datetime
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
    df.set_index('Date', inplace=True)
    
    # Створення об'єкту даних Backtrader з pandas DataFrame
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Індекс DataFrame вже містить дати
        open=df.columns.get_loc('Open'),
        high=df.columns.get_loc('High'),
        low=df.columns.get_loc('Low'),
        close=df.columns.get_loc('Close'),
        volume=df.columns.get_loc('Volume'),
        openinterest=-1  # Немає open interest
    )
    
    cerebro.adddata(data)
    
    # Встановлення початкового капіталу
    cerebro.broker.setcash(cash)
    
    # Встановлення комісії
    cerebro.broker.setcommission(commission=commission)
    
    # Додавання аналізаторів
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(btanalyzers.PyFolio, _name='pyfolio')
    
    # Початковий портфель
    print(f'Початковий капітал: ${cash:.2f}')
    
    # Запуск бектесту
    results = cerebro.run()
    strategy = results[0]
    
    # Кінцевий портфель
    final_portfolio = cerebro.broker.getvalue()
    print(f'Кінцевий капітал: ${final_portfolio:.2f}')
    print(f'Прибуток/збиток: ${final_portfolio - cash:.2f}')
    print(f'Прибуток/збиток (%): {(final_portfolio / cash - 1) * 100:.2f}%')
    
    return cerebro, strategy

# Виконання бектесту
cerebro, strategy = run_backtest('apple_data.csv', EnhancedAdaptiveStrategy)

# Візуалізація результатів бектесту
plt.figure(figsize=(16, 10))
cerebro.plot(style='candlestick', numfigs=3, barup='green', bardown='red', 
             volup='green', voldown='red', grid=True, subtxtsize=7)
plt.savefig('backtest_results.png')
plt.close()

# -----------------------------------------------------------------------------
# 5. POST-TRADE ANALYSIS
# -----------------------------------------------------------------------------

# Аналіз результатів торгівлі
def analyze_backtest_results(strategy):
    """
    Аналіз результатів бектесту
    
    Параметри:
    ----------
    strategy : bt.Strategy
        Екземпляр стратегії після бектесту
    """
    # Отримання результатів аналізаторів
    sharpe_ratio = strategy.analyzers.sharpe_ratio.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()
    
    # Виведення результатів
    print("\n===== РЕЗУЛЬТАТИ БЕКТЕСТУ =====")
    print(f"Коефіцієнт Шарпа: {sharpe_ratio['sharperatio']:.4f}")
    print(f"Максимальна просадка: {drawdown['max']['drawdown']:.2f}%")
    print(f"Максимальна просадка (грошова): ${drawdown['max']['moneydown']:.2f}")
    print(f"Річна доходність: {returns['ravg'] * 100:.2f}%")
    print(f"Загальна доходність: {returns['rtot'] * 100:.2f}%")
    
    # Аналіз угод
    if trades.get('total'):
        total_trades = trades['total']['total']
        print(f"\nЗагальна кількість угод: {total_trades}")
        
        if trades.get('won'):
            won_trades = trades['won']['total']
            win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0
            print(f"Прибуткових угод: {won_trades} ({win_rate:.2f}%)")
            print(f"Середній прибуток на угоду: ${trades['won']['pnl']['average']:.2f}")
            
        if trades.get('lost'):
            lost_trades = trades['lost']['total']
            print(f"Збиткових угод: {lost_trades} ({lost_trades / total_trades * 100 if total_trades > 0 else 0:.2f}%)")
            print(f"Середній збиток на угоду: ${trades['lost']['pnl']['average']:.2f}")
        
        if trades.get('pnl'):
            print(f"Загальний PnL: ${trades['pnl']['net']['total']:.2f}")
            print(f"Середній PnL на угоду: ${trades['pnl']['net']['average']:.2f}")
    
    # SQN (System Quality Number)
    print(f"\nSQN: {sqn['sqn']:.4f}")
    print(f"SQN Рейтинг: {sqn.get('description', 'N/A')}")
    
    # Візуалізація розподілу прибутків/збитків
    if trades.get('pnl') and trades['pnl'].get('net') and trades['pnl']['net'].get('histogram'):
        pnl_hist = trades['pnl']['net']['histogram']
        pnl_values = list(pnl_hist.keys())
        pnl_counts = list(pnl_hist.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(pnl_values, pnl_counts, alpha=0.7, color='blue')
        plt.title('Розподіл прибутків/збитків на угоду')
        plt.xlabel('PnL ($)')
        plt.ylabel('Кількість угод')
        plt.grid(True, alpha=0.3)
        plt.savefig('pnl_distribution.png')
        plt.close()

# Аналіз результатів бектесту
analyze_backtest_results(strategy)

# Візуалізація сигналів купівлі/продажу
def visualize_signals(data, buy_signals, sell_signals):
    """
    Візуалізація сигналів купівлі/продажу на графіку ціни
    
    Параметри:
    ----------
    data : pd.DataFrame
        Датафрейм з ціновими даними
    buy_signals : list
        Список кортежів (індекс, ціна) для сигналів купівлі
    sell_signals : list
        Список кортежів (індекс, ціна) для сигналів продажу
    """
    # Конвертація індексів у дати
    buy_dates = [data.index[idx] for idx, _ in buy_signals]
    buy_prices = [price for _, price in buy_signals]
    
    sell_dates = [data.index[idx] for idx, _ in sell_signals]
    sell_prices = [price for _, price in sell_signals]
    
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Close'], label='Ціна закриття AAPL', linewidth=1.5)
    
    # Відображення сигналів
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Сигнал купівлі')
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Сигнал продажу')
    
    plt.title('Сигнали купівлі/продажу для AAPL')
    plt.xlabel('Дата')
    plt.ylabel('Ціна ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trading_signals.png')
    plt.close()

# Візуалізація сигналів (припускаємо, що стратегія зберігає сигнали)
visualize_signals(data, strategy.buy_signals, strategy.sell_signals)

# Аналіз щомісячних прибутків/збитків
def analyze_monthly_returns(strategy):
    """
    Аналіз щомісячних прибутків/збитків
    
    Параметри:
    ----------
    strategy : bt.Strategy
        Екземпляр стратегії після бектесту
    """
    # Отримання аналізу pyfolio
    pyfolio = strategy.analyzers.pyfolio.get_analysis()
    
    # Перевірка наявності необхідних ключів
    if not all(key in pyfolio for key in ['returns']):
        print("Попередження: PyFolio аналізатор не повернув необхідних даних для аналізу. Використовуємо щоденні дані.")
        # Створюємо штучний датасет з історії торгівлі
        dates = strategy.datas[0].datetime.array
        prices = strategy.datas[0].close.array
        
        # Конвертуємо Backtrader datetime в pandas datetime
        dates = [bt.num2date(date) for date in dates]
        
        # Створюємо серію денних цін
        prices_series = pd.Series(prices, index=dates)
        
        # Обчислюємо денні прибутки
        returns = prices_series.pct_change().dropna()
    else:
        # Отримання щоденних прибутків з аналізатора
        returns_values = pyfolio['returns']
        
        # Якщо datetime ключ відсутній, використовуємо дати з даних стратегії
        if 'datetime' not in pyfolio:
            dates = strategy.datas[0].datetime.array
            dates = [bt.num2date(date) for date in dates[-len(returns_values):]]
        else:
            dates = pyfolio['datetime']
        
        # Перетворення в pandas серію з датами
        returns = pd.Series(returns_values, index=dates)
    
    # Обчислення щомісячних прибутків
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Створення гістограми щомісячних прибутків
    plt.figure(figsize=(14, 8))
    monthly_returns.plot(kind='bar', color=monthly_returns.apply(lambda x: 'green' if x > 0 else 'red'))
    plt.title('Щомісячні прибутки/збитки (%)')
    plt.xlabel('Місяць')
    plt.ylabel('Прибуток/збиток (%)')
    plt.grid(True, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    plt.savefig('monthly_returns.png')
    plt.close()
    
    # Створення теплової карти щомісячних прибутків
    if len(monthly_returns) > 0:
        try:
            monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(monthly_returns_pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
            plt.title('Теплова карта щомісячних прибутків (%)')
            plt.xlabel('Місяць')
            plt.ylabel('Рік')
            plt.tight_layout()
            plt.savefig('monthly_returns_heatmap.png')
            plt.close()
        except Exception as e:
            print(f"Попередження: Не вдалося створити теплову карту: {e}")
    
    return monthly_returns

# Аналіз щомісячних прибутків
monthly_returns = analyze_monthly_returns(strategy)

# Оцінка ефективності стратегії
def evaluate_strategy_performance(strategy):
    """
    Комплексна оцінка ефективності стратегії
    
    Параметри:
    ----------
    strategy : bt.Strategy
        Екземпляр стратегії після бектесту
        
    Повертає:
    ---------
    performance_metrics : dict
        Словник метрик ефективності
    """
    # Отримання аналізу pyfolio
    pyfolio = strategy.analyzers.pyfolio.get_analysis()
    
    # Перевірка наявності необхідних ключів
    if not all(key in pyfolio for key in ['returns']):
        print("Попередження: PyFolio аналізатор не повернув необхідних даних для аналізу. Використовуємо щоденні дані.")
        # Створюємо штучний датасет з історії торгівлі
        dates = strategy.datas[0].datetime.array
        prices = strategy.datas[0].close.array
        
        # Конвертуємо Backtrader datetime в pandas datetime
        dates = [bt.num2date(date) for date in dates]
        
        # Створюємо серію денних цін
        prices_series = pd.Series(prices, index=dates)
        
        # Обчислюємо денні прибутки
        returns = prices_series.pct_change().dropna()
    else:
        # Отримання щоденних прибутків з аналізатора
        returns_values = pyfolio['returns']
        
        # Якщо datetime ключ відсутній, використовуємо дати з даних стратегії
        if 'datetime' not in pyfolio:
            dates = strategy.datas[0].datetime.array
            dates = [bt.num2date(date) for date in dates[-len(returns_values):]]
        else:
            dates = pyfolio['datetime']
        
        # Перетворення в pandas серію з датами
        returns = pd.Series(returns_values, index=dates)
    
    # Обчислення загальних метрик
    cumulative_returns = (1 + returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]
    
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    daily_std = returns.std()
    annualized_std = daily_std * (252 ** 0.5)
    
    # Коефіцієнт Шарпа (використовуємо результат аналізатора для точності)
    sharpe_ratio = strategy.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    
    # Максимальна просадка
    drawdown = strategy.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown['max']['drawdown']
    
    # Сортіно (відношення прибутку до негативної волатильності)
    downside_returns = returns[returns < 0]
    sortino_ratio = (annualized_return / (downside_returns.std() * (252 ** 0.5))) if len(downside_returns) > 0 else float('inf')
    
    # Коефіцієнт Калмара (річна прибутковість / максимальна просадка)
    calmar_ratio = annualized_return / (max_drawdown / 100) if max_drawdown > 0 else float('inf')
    
    # Обчислення бета і альфа до ринку (використовуємо S&P 500 як орієнтир)
    # Для бектесту це можна обчислити окремо, якщо в нас є дані по ринку
    
    # Обчислення статистики угод
    trades_analysis = strategy.analyzers.trades.get_analysis()
    
    # Збір усіх метрик
    performance_metrics = {
        'total_return': total_return * 100,
        'annualized_return': annualized_return * 100,
        'annualized_volatility': annualized_std * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'trades_count': trades_analysis.get('total', {}).get('total', 0),
        'win_rate': trades_analysis.get('won', {}).get('total', 0) / trades_analysis.get('total', {}).get('total', 1) * 100 if trades_analysis.get('total', {}).get('total', 0) > 0 else 0,
        'profit_factor': abs(trades_analysis.get('won', {}).get('pnl', {}).get('total', 0) / trades_analysis.get('lost', {}).get('pnl', {}).get('total', -1)) if trades_analysis.get('lost', {}).get('pnl', {}).get('total', 0) != 0 else float('inf'),
        'avg_trade_duration': trades_analysis.get('len', {}).get('average', 0)
    }
    
    # Візуалізація кривої капіталу
    plt.figure(figsize=(14, 8))
    (cumulative_returns * 100).plot()
    plt.title('Крива капіталу (%)')
    plt.xlabel('Дата')
    plt.ylabel('Кумулятивна доходність (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('equity_curve.png')
    plt.close()
    
    # Візуалізація просадки
    try:
        if 'drawdown' in pyfolio:
            drawdown_series = pd.Series(pyfolio['drawdown'], index=returns.index)
            plt.figure(figsize=(14, 6))
            (drawdown_series * 100).plot()
            plt.title('Просадка (%)')
            plt.xlabel('Дата')
            plt.ylabel('Просадка (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('drawdown_curve.png')
            plt.close()
    except Exception as e:
        print(f"Попередження: Не вдалося створити графік просадки: {e}")
    
    # Виведення таблиці метрик
    print("\n===== МЕТРИКИ ЕФЕКТИВНОСТІ СТРАТЕГІЇ =====")
    for metric, value in performance_metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
    
    return performance_metrics

# Оцінка ефективності стратегії
performance_metrics = evaluate_strategy_performance(strategy)

# Порівняння з пасивною стратегією Buy & Hold
def compare_with_buy_and_hold(data, strategy):
    """
    Порівняння стратегії з пасивною стратегією Buy & Hold
    
    Параметри:
    ----------
    data : pd.DataFrame
        Датафрейм з ціновими даними
    strategy : bt.Strategy
        Екземпляр стратегії після бектесту
    """
    # Отримання аналізу pyfolio
    pyfolio = strategy.analyzers.pyfolio.get_analysis()
    
    # Перевірка наявності необхідних ключів
    if not all(key in pyfolio for key in ['returns']):
        print("Попередження: PyFolio аналізатор не повернув необхідних даних для аналізу. Використовуємо щоденні дані.")
        # Створюємо штучний датасет з історії торгівлі
        dates = strategy.datas[0].datetime.array
        prices = strategy.datas[0].close.array
        
        # Конвертуємо Backtrader datetime в pandas datetime
        dates = [bt.num2date(date) for date in dates]
        
        # Створюємо серію денних цін
        prices_series = pd.Series(prices, index=dates)
        
        # Обчислюємо денні прибутки
        strategy_returns = prices_series.pct_change().dropна()
    else:
        # Отримання щоденних прибутків з аналізатора
        returns_values = pyfolio['returns']
        
        # Якщо datetime ключ відсутній, використовуємо дати з даних стратегії
        if 'datetime' not in pyfolio:
            dates = strategy.datas[0].datetime.array
            dates = [bt.num2date(date) for date in dates[-len(returns_values):]]
        else:
            dates = pyfolio['datetime']
        
        # Перетворення в pandas серію з датами
        strategy_returns = pd.Series(returns_values, index=dates)
    
    # Обчислення прибутків для Buy & Hold
    buy_hold_returns = data['Returns'].dropна()
    
    # Переіндексація, щоб відповідати індексу стратегії
    buy_hold_returns = buy_hold_returns.reindex(strategy_returns.index)
    
    # Обчислення кумулятивних прибутків
    strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
    buy_hold_cum_returns = (1 + buy_hold_returns).cumprod() - 1
    
    # Візуалізація порівняння
    plt.figure(figsize=(14, 8))
    (strategy_cum_returns * 100).plot(label='Стратегія', linewidth=2)
    (buy_hold_cum_returns * 100).plot(label='Buy & Hold', linewidth=2)
    plt.title('Порівняння стратегії з Buy & Hold')
    plt.xlabel('Дата')
    plt.ylabel('Кумулятивна доходність (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_vs_buyhold.png')
    plt.close()
    
    # Обчислення підсумкових метрик
    strategy_total_return = strategy_cum_returns.iloc[-1] * 100
    buy_hold_total_return = buy_hold_cum_returns.iloc[-1] * 100
    
    strategy_annual_return = (1 + strategy_cum_returns.iloc[-1]) ** (252 / len(strategy_returns)) - 1
    buy_hold_annual_return = (1 + buy_hold_cum_returns.iloc[-1]) ** (252 / len(buy_hold_returns)) - 1
    
    strategy_volatility = strategy_returns.std() * (252 ** 0.5) * 100
    buy_hold_volatility = buy_hold_returns.std() * (252 ** 0.5) * 100
    
    # Виведення порівняльної таблиці
    print("\n===== ПОРІВНЯННЯ З BUY & HOLD =====")
    print(f"{'':20} {'Стратегія':15} {'Buy & Hold':15}")
    print(f"{'-'*50}")
    print(f"{'Загальна доходність':20} {strategy_total_return:15.2f}% {buy_hold_total_return:15.2f}%")
    print(f"{'Річна доходність':20} {strategy_annual_return*100:15.2f}% {buy_hold_annual_return*100:15.2f}%")
    print(f"{'Волатильність':20} {strategy_volatility:15.2f}% {buy_hold_volatility:15.2f}%")
    
    # Обчислення альфи і бети
    # Бета - чутливість стратегії до ринку
    covariance = np.cov(strategy_returns, buy_hold_returns)[0, 1]
    variance = np.var(buy_hold_returns)
    beta = covariance / variance if variance != 0 else 0
    
    # Альфа - надлишкова доходність стратегії над ринком з урахуванням ризику
    risk_free_rate = 0.02 / 252  # Припускаємо 2% річної безризикової ставки
    alpha = (strategy_annual_return - risk_free_rate) - beta * (buy_hold_annual_return - risk_free_rate)
    
    print(f"{'Бета':20} {beta:15.4f}")
    print(f"{'Альфа (річна)':20} {alpha*100:15.2f}%")
    
    return {
        'strategy_return': strategy_total_return,
        'buy_hold_return': buy_hold_total_return,
        'strategy_annual_return': strategy_annual_return * 100,
        'buy_hold_annual_return': buy_hold_annual_return * 100,
        'strategy_volatility': strategy_volatility,
        'buy_hold_volatility': buy_hold_volatility,
        'beta': beta,
        'alpha': alpha * 100
    }

# Отримання прибутків стратегії з pyfolio аналізатора та порівняння з Buy & Hold
comparison_metrics = compare_with_buy_and_hold(data, strategy)

# Висновки та рекомендації
print("\n===== ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ =====")
print(f"1. Стратегія показала {'кращі' if comparison_metrics['strategy_return'] > comparison_metrics['buy_hold_return'] else 'гірші'} результати порівняно з пасивною стратегією Buy & Hold.")
print(f"2. Відношення прибутку до ризику: {performance_metrics['sharpe_ratio']:.2f} (Шарп).")
print(f"3. Максимальна просадка капіталу: {performance_metrics['max_drawdown']:.2f}%.")
print(f"4. Вінрейт стратегії: {performance_metrics['win_rate']:.2f}%.")

if performance_metrics['sharpe_ratio'] > 1:
    print("5. Стратегія показує хороше співвідношення прибутку до ризику.")
else:
    print("5. Рекомендується поліпшити співвідношення прибутку до ризику.")

if performance_metrics['max_drawdown'] > 20:
    print("6. Необхідно оптимізувати управління ризиками для зменшення максимальної просадки.")
else:
    print("6. Система управління ризиками працює ефективно.")

if performance_metrics['win_rate'] < 50:
    print("7. Необхідно поліпшити точність торгових сигналів для підвищення відсотка прибуткових угод.")
else:
    print("7. Стратегія показує хороший відсоток прибуткових угод.")

# Збереження останнього графіка - порівняння з іншими стратегіями або оптимізованими параметрами
# Це можна було б зробити при бажанні, запустивши стратегію з різними параметрами

print("\n===== ЗАВЕРШЕННЯ АНАЛІЗУ =====")
print("Всі результати аналізу збережені у вигляді графіків.")