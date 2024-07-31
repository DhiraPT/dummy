# Trading RL: Reinforcement Learning for Trading

This project aims to apply reinforcement learning techniques to develop trading strategies. By leveraging the power of machine learning algorithms, we seek to create intelligent trading agents that can make informed decisions in dynamic market environments.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Reinforcement Learning Approach](#reinforcement-learning-approach)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the world of financial markets, trading decisions need to be made quickly and accurately to maximize profits. Traditional trading strategies often rely on human intuition and experience, which can be limited and prone to biases. Reinforcement learning offers a promising approach to automate trading decisions by training agents to learn from their interactions with the market.

This project aims to explore and implement various reinforcement learning algorithms, such as Q-learning and Deep Q-Networks (DQN), to develop robust and adaptive trading strategies. By utilizing historical market data and real-time information, our goal is to create intelligent agents that can adapt to changing market conditions and make profitable trading decisions.

## Installation

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/DhiraPT/trading-rl.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set up your trading environment and data sources.
4. Run the main script: `python main.py`

## Reinforcement Learning Approach

### Observation Space
OHLCV (Open, High, Low, Close, Volume) data and other indicators for each asset across 1-minute, 5-minute, 15-minute, 30-minute, and 60-minute timeframes from past candles. The number of past candles is determined by the specified window size, excluding the current candle.

### Action Space
A list of numbers ranging from -1 to 1 (inclusive), where each number represents an action for a specific asset in our portfolio watchlist.
- **Sign:**
  - Positive: Buy
  - Zero: Hold
  - Negative: Sell
- **Magnitude:** Represents the fraction of the asset being bought or sold.
  - For buying actions, the magnitude is based on the total balance left after accounting for the fees associated with buying.
  - For selling actions, the magnitude is based on the total positions held.

### Execution
- Buy: At the open price of the current candle.
- Sell: At the open price of the current candle.

### Reward Calculation
The reward is calculated as the weighted sum of the closing prices of the current candle across each timeframe.

## Contributing

Contributions to this project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Let's collaborate and improve the performance and usability of our trading agents together.

## License

This project is licensed under the [MIT License](LICENSE).
