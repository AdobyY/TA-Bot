## Анотація
"""
Розроблено багатофакторну торгову стратегію для криптовалют з використанням бібліотеки backtrader. 
Стратегія поєднує технічний аналіз, волатильність, об'єми торгів та модель машинного навчання для прийняття торгових рішень. 
Стратегія тестується на історичних даних декількох криптовалют.
"""

# 1. Отримання даних та розрахунок базових індикаторів

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ccxt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Налаштування графіків matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

# Функція для отримання даних криптовалют
def get_crypto_data(symbol, timeframe='1d', start_date=None, end_date=None):
    """
    Отримання історичних даних для криптовалюти через CCXT
    
    Параметри:
    symbol (str): Символ криптовалюти (наприклад, 'BTC/USDT')
    timeframe (str): Таймфрейм даних ('1d' для денних даних)
    start_date (str): Початкова дата у форматі 'YYYY-MM-DD'
    end_date (str): Кінцева дата у форматі 'YYYY-MM-DD'
    
    Повертає:
    pandas.DataFrame: Історичні дані криптовалюти
    """
    # Ініціалізація біржі (використовуємо Binance)
    exchange = ccxt.binance()
    
    # Конвертація дат в timestamp
    since = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    # Отримання даних
    all_candles = []
    while since < end_timestamp:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Створення DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

# Визначення часового періоду для аналізу
start_date = '2020-01-01'
end_date = '2023-12-31'

# Отримання даних для кількох криптовалют
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
data_dict = {}

for symbol in symbols:
    print(f"Завантаження даних для {symbol}...")
    data_dict[symbol] = get_crypto_data(symbol, start_date=start_date, end_date=end_date)

# Розрахунок базових індикаторів для Bitcoin (BTC/USDT)
btc_data = data_dict['BTC/USDT']

# Розрахунок ковзних середніх
btc_data['SMA20'] = btc_data['Close'].rolling(window=20).mean()
btc_data['SMA50'] = btc_data['Close'].rolling(window=50).mean()
btc_data['SMA200'] = btc_data['Close'].rolling(window=200).mean()

# Розрахунок експоненціальних ковзних середніх
btc_data['EMA12'] = btc_data['Close'].ewm(span=12, adjust=False).mean()
btc_data['EMA26'] = btc_data['Close'].ewm(span=26, adjust=False).mean()

# Розрахунок MACD
btc_data['MACD'] = btc_data['EMA12'] - btc_data['EMA26']
btc_data['Signal_Line'] = btc_data['MACD'].ewm(span=9, adjust=False).mean()
btc_data['MACD_Histogram'] = btc_data['MACD'] - btc_data['Signal_Line']

# Розрахунок RSI (Relative Strength Index)
delta = btc_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
btc_data['RSI'] = 100 - (100 / (1 + rs))

# Розрахунок ATR (Average True Range) для вимірювання волатильності
high_low = btc_data['High'] - btc_data['Low']
high_close = np.abs(btc_data['High'] - btc_data['Close'].shift())
low_close = np.abs(btc_data['Low'] - btc_data['Close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
btc_data['ATR'] = true_range.rolling(window=14).mean()

# Розрахунок відносного об'єму (нормалізований за 20-денним середнім)
btc_data['Volume_SMA20'] = btc_data['Volume'].rolling(window=20).mean()
btc_data['Relative_Volume'] = btc_data['Volume'] / btc_data['Volume_SMA20']

# Розрахунок Bolling Bands
btc_data['BB_Middle'] = btc_data['Close'].rolling(window=20).mean()
btc_data['BB_StdDev'] = btc_data['Close'].rolling(window=20).std()
btc_data['BB_Upper'] = btc_data['BB_Middle'] + (btc_data['BB_StdDev'] * 2)
btc_data['BB_Lower'] = btc_data['BB_Middle'] - (btc_data['BB_StdDev'] * 2)

# Підготовка даних для візуалізації
btc_data = btc_data.dropna()

# Візуалізація ціни закриття та ковзних середніх
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(btc_data.index, btc_data['Close'], label='Ціна закриття')
plt.plot(btc_data.index, btc_data['SMA20'], label='SMA20')
plt.plot(btc_data.index, btc_data['SMA50'], label='SMA50')
plt.plot(btc_data.index, btc_data['SMA200'], label='SMA200')
plt.plot(btc_data.index, btc_data['BB_Upper'], 'r--', label='BB Upper')
plt.plot(btc_data.index, btc_data['BB_Lower'], 'g--', label='BB Lower')
plt.title('BTC/USDT - Ціна акції та індикатори')
plt.ylabel('Ціна')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(btc_data.index, btc_data['MACD'], label='MACD')
plt.plot(btc_data.index, btc_data['Signal_Line'], label='Сигнальна лінія')
plt.bar(btc_data.index, btc_data['MACD_Histogram'], label='MACD Гістограма')
plt.title('MACD')
plt.ylabel('Значення')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(btc_data.index, btc_data['RSI'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', label='Перекуплений (70)')
plt.axhline(y=30, color='g', linestyle='--', label='Перепроданий (30)')
plt.title('RSI')
plt.ylabel('Значення')
plt.legend()

plt.tight_layout()

# 2. Передторговий аналіз - Кореляційний аналіз між активами

# Створення DataFrame з цінами закриття для всіх акцій
close_prices = pd.DataFrame()
for ticker, data in data_dict.items():
    close_prices[ticker] = data['Close']

# Розрахунок кореляційної матриці
correlation_matrix = close_prices.corr()

# Візуалізація кореляційної матриці
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Кореляційна матриця цін закриття')

# Аналіз розподілу дохідності
returns = close_prices.pct_change().dropna()

plt.figure(figsize=(15, 10))
for i, ticker in enumerate(symbols):
    plt.subplot(len(symbols), 1, i+1)
    plt.hist(returns[ticker], bins=50, alpha=0.7)
    plt.title(f'Розподіл дохідності для {ticker}')
    plt.ylabel('Частота')
    
plt.tight_layout()

# 3. Розробка торгових сигналів - Побудова моделі машинного навчання для прогнозування напрямку ціни

# Функція для створення навчальних даних
def create_features(data, window_size=10):
    """
    Створення функцій для моделі машинного навчання на основі технічних індикаторів
    
    Параметри:
    data (DataFrame): Історичні дані акції з розрахованими індикаторами
    window_size (int): Розмір вікна для розрахунку функцій
    
    Повертає:
    tuple: (X - матриця функцій, y - цільові значення)
    """
    # Визначення ознак для моделі
    features = [
        'SMA20', 'SMA50', 'SMA200', 
        'MACD', 'Signal_Line', 'MACD_Histogram',
        'RSI', 'ATR', 'Relative_Volume',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_StdDev'
    ]
    
    # Створення матриці функцій
    X = data[features].copy()
    
    # Додавання відносних ознак
    X['SMA20_50_Ratio'] = data['SMA20'] / data['SMA50']
    X['SMA50_200_Ratio'] = data['SMA50'] / data['SMA200']
    X['Close_SMA20_Ratio'] = data['Close'] / data['SMA20']
    X['Close_SMA50_Ratio'] = data['Close'] / data['SMA50']
    X['Close_SMA200_Ratio'] = data['Close'] / data['SMA200']
    
    # Розрахунок процентної зміни для кожного показника
    for feature in features:
        X[f'{feature}_Change'] = data[feature].pct_change(periods=1)
        
    # Додавання лагових змінних
    for lag in range(1, window_size + 1):
        X[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        X[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        X[f'Return_Lag_{lag}'] = data['Close'].pct_change(periods=lag)
        
    # Розрахунок волатильності за різні періоди
    X['Volatility_5'] = data['Close'].pct_change().rolling(window=5).std()
    X['Volatility_10'] = data['Close'].pct_change().rolling(window=10).std()
    X['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Визначення цільової змінної (1 якщо ціна зросла через 5 днів, 0 якщо впала)
    y = (data['Close'].shift(-5) > data['Close']).astype(int)
    
    # Видалення рядків з відсутніми значеннями
    X = X.dropna()
    y = y.loc[X.index]
    
    return X, y

# Створення функцій і цільових значень для BTC/USDT
X, y = create_features(btc_data)

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Нормалізація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання моделі Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Перевірка продуктивності моделі
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Точність моделі: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Візуалізація важливості ознак
feature_importance = model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(12, 10))
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.title('Важливість ознак для моделі Random Forest')
plt.xlabel('Важливість')
plt.ylabel('Ознака')

plt.tight_layout()

# 4. Виконання торгів - Імплементація стратегії в backtrader

# Визначення кастомного індикатора для результатів моделі машинного навчання
class MLSignal(bt.Indicator):
    """
    Кастомний індикатор для сигналів машинного навчання
    """
    lines = ('prediction',)
    params = (('model', None), ('scaler', None), ('feature_names', None))
    
    def __init__(self):
        self.data_window = {}
        for feature in self.p.feature_names:
            self.data_window[feature] = []
    
    def next(self):
        # В реальному середовищі тут би відбувалося створення ознак і передбачення
        # Для спрощення прикладу, ми використовуємо випадкові передбачення
        self.lines.prediction[0] = np.random.choice([0, 1], p=[0.5, 0.5])

# Визначення основної стратегії
class MultifactorMLStrategy(bt.Strategy):
    """
    Багатофакторна стратегія для криптовалют з використанням машинного навчання
    """
    params = (
        ('ml_signal_threshold', 0.6),  # Поріг ймовірності для сигналу від ML
        ('sma_short', 10),            # Коротка ковзна середня (зменшено для криптовалют)
        ('sma_long', 30),             # Довга ковзна середня (зменшено для криптовалют)
        ('rsi_period', 14),           # Період RSI
        ('rsi_overbought', 75),       # Поріг перекупленості RSI
        ('rsi_oversold', 25),         # Поріг перепроданості RSI
        ('macd_fast', 12),            # Швидкий період MACD
        ('macd_slow', 26),            # Повільний період MACD
        ('macd_signal', 9),           # Сигнальний період MACD
        ('atr_period', 14),           # Період ATR
        ('bb_period', 20),            # Період смуг Боллінджера
        ('bb_dev', 2.0),              # Стандартне відхилення смуг Боллінджера
        ('vol_period', 20),           # Період для об'єму
        ('risk_per_trade', 0.02),     # Ризик на угоду (2% від капіталу)
        ('trailing_stop', True),      # Використовувати трейлінг-стоп
        ('trailing_percent', 0.02),   # Відсоток для трейлінг-стопу (2%)
        ('take_profit', 0.05),        # Рівень для take profit (5%)
        ('portfolio_allocation', 0.8), # Максимальний відсоток портфеля для використання
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # Зберігаємо посилання на індикатори, які будемо використовувати
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # Створення технічних індикаторів
        # Ковзні середні
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.dataclose, period=self.params.sma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.dataclose, period=self.params.sma_long)
        self.sma_200 = bt.indicators.SimpleMovingAverage(
            self.dataclose, period=200)
        
        # Індикатор перетину ковзних середніх
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        
        # RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period)
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.dataclose,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal)
        
        # ATR для визначення волатильності та стоп-лосів
        self.atr = bt.indicators.ATR(
            self.datas[0], period=self.params.atr_period)
        
        # Смуги Боллінджера
        self.bollinger = bt.indicators.BollingerBands(
            self.dataclose, period=self.params.bb_period, devfactor=self.params.bb_dev)
        
        # Відносний об'єм
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.datavolume, period=self.params.vol_period)
        
        # Додаємо кастомний індикатор для ML сигналів
        self.ml_signal = MLSignal(model=None, scaler=None, feature_names=[])
        
        # Для відстеження відкритих позицій і їх стоп-лосів
        self.orders = {}  # словник для зберігання стоп-ордерів
        self.trade_history = []  # історія торгів
        self.current_positions = {}  # поточні позиції
        
        # Для розрахунку розміру позиції
        self.position_size = 0
        
        # Змінні для трейлінг-стопу
        self.trailing_stop_price = None
        self.take_profit_price = None
        
        # Лічильник для врахування часу в ринку
        self.in_market_days = 0
    
    def notify_order(self, order):
        """
        Отримуємо повідомлення про зміни в ордерах
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Замовлення прийнято - нічого не робимо
            return
        
        # Перевіряємо, чи замовлення виконано
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                # Розрахунок рівнів стоп-лосу та тейк-профіту
                self.trailing_stop_price = order.executed.price * (1 - self.params.trailing_percent)
                self.take_profit_price = order.executed.price * (1 + self.params.take_profit)
                
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Скидаємо посилання на замовлення
        self.order = None
    
    def notify_trade(self, trade):
        """
        Отримуємо повідомлення про завершені торги
        """
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
        # Додаємо інформацію про угоду до історії
        trade_info = {
            'date_open': self.data.datetime.date(-trade.barlen).isoformat(),
            'date_close': self.data.datetime.date(0).isoformat(),
            'price_open': trade.price,
            'price_close': trade.pnlcomm / trade.size + trade.price,
            'size': trade.size,
            'pnl': trade.pnlcomm,
            'duration': trade.barlen
        }
        self.trade_history.append(trade_info)
        
        # Скидаємо лічильник часу на ринку
        self.in_market_days = 0
    
    def calculate_position_size(self):
        """
        Розрахунок розміру позиції на основі ATR та параметрів ризику
        """
        # Визначаємо ризик на угоду
        risk_amount = self.broker.getvalue() * self.params.risk_per_trade
        
        # Використовуємо ATR для встановлення стоп-лосу
        stop_loss_size = self.atr[0] * 1.5  # Зменшено множник для криптовалют
        
        # Розрахунок кількості одиниць для купівлі
        if stop_loss_size > 0:
            size = risk_amount / stop_loss_size
            # Округлюємо до 4 знаків після коми для криптовалют
            return round(size, 4)
        else:
            return 0
    
    def should_buy(self):
        """
        Визначення умов для купівлі
        """
        # Комбінація сигналів для прийняття рішення про купівлю
        
        # 1. Перевірка перетину ковзних середніх
        sma_signal = self.crossover > 0
        
        # 2. RSI показує, що ринок перепроданий
        rsi_signal = self.rsi < self.params.rsi_oversold
        
        # 3. MACD показує бичачу дивергенцію
        macd_signal = self.macd.macd > self.macd.signal
        
        # 4. Ціна знаходиться нижче нижньої смуги Боллінджера
        bb_signal = self.dataclose[0] < self.bollinger.lines.bot[0]
        
        # 5. Підвищений обсяг торгів порівняно з середнім
        volume_signal = self.datavolume[0] > self.vol_sma[0] * 1.5
        
        # Комбінація сигналів (потрібно щонайменше 3 з 5 для підтвердження)
        signals = [sma_signal, rsi_signal, macd_signal, bb_signal, volume_signal]
        signal_count = sum(signals)
        
        # Додаткова перевірка ризику - чи не перевищили ми ліміт алокації портфеля
        current_allocation = self.broker.getvalue() / self.broker.getvalue() * 100
        allocation_ok = current_allocation < self.params.portfolio_allocation * 100
        
        return signal_count >= 3 and allocation_ok
    
    def should_sell(self):
        """
        Визначення умов для продажу
        """
        # 1. Перетин ковзних середніх (мертвий хрест)
        sma_signal = self.crossover < 0
        
        # 2. RSI показує, що ринок перекуплений
        rsi_signal = self.rsi > self.params.rsi_overbought
        
        # 3. MACD показує ведмежу дивергенцію
        macd_signal = self.macd.macd < self.macd.signal
        
        # 4. Ціна знаходиться вище верхньої смуги Боллінджера
        bb_signal = self.dataclose[0] > self.bollinger.lines.top[0]
        
        # 5. Знижений обсяг торгів порівняно з середнім
        volume_signal = self.datavolume[0] < self.vol_sma[0] * 0.5
        
        # Комбінація сигналів (потрібно щонайменше 3 з 5 для підтвердження)
        signals = [sma_signal, rsi_signal, macd_signal, bb_signal, volume_signal]
        signal_count = sum(signals)
        
        return signal_count >= 3
    
    def update_trailing_stop(self):
        """
        Оновлення трейлінг-стопу
        """
        if not self.position or not self.params.trailing_stop:
            return
        
        # Якщо поточна ціна вища за попередню ціну трейлінг-стопу
        if self.dataclose[0] > self.trailing_stop_price:
            # Розрахунок нового трейлінг-стопу
            new_stop = self.dataclose[0] * (1 - self.params.trailing_percent)
            
            # Оновлюємо стоп тільки якщо новий стоп вищий за попередній
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                self.log(f'Trailing stop updated to: {self.trailing_stop_price:.2f}')
    
    def check_exit_conditions(self):
        """
        Перевірка умов виходу з позиції
        """
        if not self.position:
            return False
        
        # Перевірка стоп-лосу
        if self.dataclose[0] < self.trailing_stop_price:
            self.log(f'Triggering STOP LOSS at {self.dataclose[0]:.2f}, stop: {self.trailing_stop_price:.2f}')
            return True
        
        # Перевірка тейк-профіту
        if self.dataclose[0] >= self.take_profit_price:
            self.log(f'Triggering TAKE PROFIT at {self.dataclose[0]:.2f}, target: {self.take_profit_price:.2f}')
            return True
        
        # Перевірка сигналу
        if self.should_sell():
            self.log(f'Triggering SELL SIGNAL at {self.dataclose[0]:.2f}')
            return True
        
        return False
    
    def next(self):
        """
        Основний метод стратегії, який викликається для кожного нового бару
        """
        # Оновлення трейлінг-стопу для відкритих позицій
        self.update_trailing_stop()
        
        # Перевірка умов виходу з позиції
        if self.position and self.check_exit_conditions():
            self.close()
            return
        
        # Перевірка умов входу в позицію
        if not self.position and self.should_buy():
            # Розрахунок розміру позиції
            size = self.calculate_position_size()
            
            if size > 0:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.buy(size=size)
                
                # Збільшуємо лічильник часу в ринку
                self.in_market_days = 0

# 5. Запуск бектестування

def run_backtest(symbol, start_date, end_date):
    """
    Запуск бектестування для однієї криптовалюти
    
    Параметри:
    symbol (str): Символ криптовалюти
    start_date (str): Початкова дата
    end_date (str): Кінцева дата
    """
    try:
        # Створення екземпляра cerebro
        cerebro = bt.Cerebro()
        
        # Додавання даних
        data = bt.feeds.PandasData(
            dataname=get_crypto_data(symbol, start_date=start_date, end_date=end_date),
            datetime=None,  # використовуємо індекс як дату
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        
        cerebro.adddata(data)
        
        # Додавання стратегії
        cerebro.addstrategy(MultifactorMLStrategy)
        
        # Налаштування брокера
        cerebro.broker.setcash(10000.0)  # Початковий капітал (зменшено для криптовалют)
        cerebro.broker.setcommission(commission=0.001)  # Комісія 0.1%
        
        # Додавання аналізаторів
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Запуск бектестування
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f} USDT')
        results = cerebro.run()
        strat = results[0]
        
        # Виведення результатів
        print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f} USDT')
        
        # Отримання та виведення Sharpe Ratio з обробкою помилок
        try:
            sharpe_ratio = strat.analyzers.sharpe.get_analysis()
            if sharpe_ratio and isinstance(sharpe_ratio, dict) and 'sharperatio' in sharpe_ratio and sharpe_ratio['sharperatio'] is not None:
                print(f'Sharpe Ratio: {sharpe_ratio["sharperatio"]:.3f}')
            else:
                print('Sharpe Ratio: N/A')
        except Exception as e:
            print(f'Sharpe Ratio: Error calculating ({str(e)})')
        
        # Отримання та виведення Drawdown з обробкою помилок
        try:
            drawdown = strat.analyzers.drawdown.get_analysis()
            if drawdown and isinstance(drawdown, dict) and 'max' in drawdown and 'drawdown' in drawdown['max'] and drawdown['max']['drawdown'] is not None:
                print(f'Max Drawdown: {drawdown["max"]["drawdown"]:.2%}')
            else:
                print('Max Drawdown: N/A')
        except Exception as e:
            print(f'Max Drawdown: Error calculating ({str(e)})')
        
        # Отримання та виведення Returns з обробкою помилок
        try:
            returns = strat.analyzers.returns.get_analysis()
            if returns and isinstance(returns, dict) and 'rnorm100' in returns and returns['rnorm100'] is not None:
                print(f'Annual Return: {returns["rnorm100"]:.2f}%')
            else:
                print('Annual Return: N/A')
        except Exception as e:
            print(f'Annual Return: Error calculating ({str(e)})')
        
        # Аналіз торгів з обробкою помилок
        try:
            trade_analysis = strat.analyzers.trades.get_analysis()
            print('\nTrade Analysis:')
            
            if hasattr(trade_analysis, 'total') and hasattr(trade_analysis.total, 'total'):
                print(f'Total Trades: {trade_analysis.total.total}')
            else:
                print('Total Trades: N/A')
            
            if hasattr(trade_analysis, 'won') and hasattr(trade_analysis.won, 'total'):
                print(f'Winning Trades: {trade_analysis.won.total}')
                if hasattr(trade_analysis.won, 'pnl') and hasattr(trade_analysis.won.pnl, 'average'):
                    print(f'Average Win: {trade_analysis.won.pnl.average:.2f} USDT')
            else:
                print('Winning Trades: N/A')
            
            if hasattr(trade_analysis, 'lost') and hasattr(trade_analysis.lost, 'total'):
                print(f'Losing Trades: {trade_analysis.lost.total}')
                if hasattr(trade_analysis.lost, 'pnl') and hasattr(trade_analysis.lost.pnl, 'average'):
                    print(f'Average Loss: {trade_analysis.lost.pnl.average:.2f} USDT')
            else:
                print('Losing Trades: N/A')
        except Exception as e:
            print(f'Trade Analysis: Error calculating ({str(e)})')
        
        # Візуалізація результатів
        cerebro.plot(style='candlestick', barup='green', bardown='red')
        
    except Exception as e:
        print(f'Error during backtesting {symbol}: {str(e)}')

if __name__ == '__main__':
    # Запуск бектестування для кожної криптовалюти
    for symbol in symbols:
        print(f'\nBacktesting {symbol}...')
        run_backtest(symbol, start_date, end_date)