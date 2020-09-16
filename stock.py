import os
import csv
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
import pandas as pd


api_key = 'KCOeXUgLebA1rsdO2lPInrIaoQAu2FmqXgvm94uDOYeB1OS1G1HsRZNxktniPQSd'

api_secret = 'iDSWnpo5j5pR67KpINvuyel3vLbWgYARshXnIpI0pZuljE2KZAIrYIfXAfwXycNg'

client = Client(api_key, api_secret)
btc_price = {'error': False}


def btc_trade_history(msg):
    ''' define how to process incoming WebSocket messages '''
    if msg['e'] != 'error':
        print(msg['c'])
        btc_price['last'] = msg['c']
        btc_price['bid'] = msg['b']
        btc_price['last'] = msg['a']
    else:
        btc_price['error'] = True


bsm = BinanceSocketManager(client)
conn_key = bsm.start_symbol_ticker_socket('BTCUSDT', btc_trade_history)

bsm.start()


timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1m')
print(timestamp)

# bsm.stop_socket(conn_key)


# request historical candle (or klines) data
# bars = client.get_historical_klines('BTCUSDT','1m', timestamp, limit=1000)

bars = client.get_historical_klines(
    "BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")


with open('btc_bars.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    for line in bars:
        wr.writerow(line)


with open('btc_bars2.csv', 'w') as d:
    for line in bars:
        d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
