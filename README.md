# Real-time intraday  bubbles trading

### Goal
Analyze intraday stock price and build trivial trading strategiesin order to observe regimes and build a strategy of strategies.

## Installation 💻
The code is optimized for Python 3.11.

### Library
The following library are used:
```bash- matplotlib~=3.8.4
seaborn~=0.13.2
numpy~=1.26.4
pandas~=2.2.2
polars~=1.8.2
tqdm~=4.66.4
regex~=2023.10.3
plotly~=5.22.0
```

## [main.py](main.py)
The main file loads the data and applies the strategies to the cleaned data.
The general setup should follow:
```bash
├── README.md
│   ├── excess_volume.py
│   ├── momentum.py
│   └── volatility_trading_strategy.py
├── data
│   ├── Raw
│   │   └── sp100_2004-8
│   │       ├── bbo
│   │       │   ├── AA.N
│   │       │   ├── ABT.N
│   │       │   ├── ...
│   │       └── trade
│   │       │   ├── AA.N
│   │       │   ├── ABT.N
│   │       │   ├── ...
│   ├── clean
│   │   ├── AA
│   │   │   ├── 2004
│   │   │   ├── 2005
│   │   │   ├── 2006
│   │   │   ├── 2007
│   │   │   └── 2008
│   │   ├── ABT
│   │   │   ├── ...
├── main.py
├── requirements.txt
├── strategy_runner.py
└── utils
    ├── data_handler_polars.py
    └── easy_plotter.py
```
## Files
- [strategy_runner.py](strategy_runner.py) : Pipeline that uses the clean data a applies the strategies in order to compute the daily returns
- [test.py](test.py) : Runs a demo project 
- [network.py](network.py) : Computes the monthly correlation networks

## Directories
[Strategies](Strategies) :
This directory regroups the various strategies we want to implement:
- [excess_volume.py](Strategies/excess_volume.py) : invest when excessive volumes are traded in the market.
- [momentum.py](Strategies/momentum.py) : Momentum strategy with a long and short window.
- [volatility_trading_strategy.py](Strategies/volatility_trading_strategy.py) : Trades when excess volatility appears in the stock.

[utils](utils) :
This file regroups some simple use functions used for the data loading as well as the graphing functions.
- [data_handler_polars.py](utils/data_handler_polars.py) : data loading functions in order to preprocess the data.
- [easy_plotter.py](utils/easy_plotter.py) : contains functions to plot graphs.

[Graphs](Graphs) :
This file contains various graphs used for our analysis.


## Usage 🫳
To run the code, make sure all the requirements are satisfied (see [requirements.txt](requirements.txt)). Then, fetch the data. You can either use the full data from SWITCHdrive (download [sp100_2008-4.tar](https://drive.switch.ch/index.php/s/0X3Je6DauQRzD2r?path=%2F)) or the demo data (smaller, only 2 tickers [data.zip](https://drive.google.com/drive/folders/1tvmYr9ElusfIkPoZieeVoobwyVXyfGHG?usp=share_link)). Then unzip the file, and organise your directory as in the tree from above. Finally, you can run the project from [main.py](main.py) and select the desired features. 

To simply have a demonstration of the project, use the small data and select ```demo_project = True``` (and everything else ```False```). This will give an overview of all the features and strategies.

## Contact 📒
- Guillaume Ferrer: guillaume[dot]ferrer[at]epfl[dot]ch
- Gustave Besacier: gustave[dot]besacier[at]epfl[dot]ch
- Edouard Bueche: edouard[dot]bueche[at]epfl[dot]ch
