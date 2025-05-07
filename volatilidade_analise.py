"""
DISCLAIMER:

Este software é fornecido apenas para fins educacionais e de pesquisa.
Não tem como objetivo fornecer aconselhamento de investimento, e nenhuma recomendação de investimento é feita aqui.
Os desenvolvedores não são consultores financeiros e não aceitam responsabilidade por quaisquer decisões financeiras
ou perdas resultantes do uso deste software. Sempre consulte um consultor financeiro profissional antes de
tomar qualquer decisão de investimento.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import os
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Union


# --- Configuração do Logging ---
LOG_FILE = "analysis_log.txt"

# Configurar o logger para registrar em arquivo e no console (para depuração)
logger = logging.getLogger("VolatilityAnalyzer")
logger.setLevel(logging.DEBUG) # Captura todos os níveis de log

# Evitar adicionar manipuladores múltiplos se o script for recarregado pelo Streamlit
if not logger.handlers:
    # Manipulador de arquivo
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8') # 'a' para append
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO) # Nível para o arquivo de log
    logger.addHandler(file_handler)

    # Manipulador de console (opcional, para depuração mais fácil durante o desenvolvimento)
    # stream_handler = logging.StreamHandler()
    # stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    # stream_handler.setFormatter(stream_formatter)
    # stream_handler.setLevel(logging.DEBUG)
    # logger.addHandler(stream_handler)

logger.info("="*50)
logger.info("Aplicação Analisador de Volatilidade Iniciada/Recarregada")
logger.info("="*50)

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Analisador de Estratégias para Mercados BR e US",
    page_icon="📈",
    layout="wide"
)

# --- API Keys ---
# Padrões, podem ser sobrescritos por variáveis de ambiente
DEFAULT_API_KEYS = {
    "alphavantage": os.getenv("ALPHAVANTAGE_API_KEY", "EOOUYAX9JC3DRX45"),
    "finnhub": os.getenv("FINNHUB_API_KEY", "d0db4ghr01qhd59vd3bgd0db4ghr01qhd59vd3c0")
}

def get_api_key(service: str) -> str:
    """Retorna a chave API do usuário (se fornecida) ou a padrão."""
    user_key = st.session_state.get(f"user_{service}_api_key", "")
    if user_key and user_key.strip():
        logger.debug(f"Usando chave API fornecida pelo usuário para {service}.")
        return user_key.strip()
    logger.debug(f"Usando chave API padrão/ambiente para {service}.")
    return DEFAULT_API_KEYS[service]

# --- Funções de Obtenção de Dados ---

@st.cache_data(ttl=3600)
def get_stock_data(symbol: str, _status_placeholder: Optional[st.empty] = None) -> pd.DataFrame:
    """
    Obter dados de ações (OHLCV ajustado) da API Alpha Vantage com fallback para yFinance.
    """
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    logger.info(f"Tentando obter dados históricos para {symbol}.")

    # Tentar Alpha Vantage
    alphavantage_key = get_api_key("alphavantage")
    if _status_placeholder:
        _status_placeholder.text(f"Buscando dados para {symbol} via Alpha Vantage...")
    logger.debug(f"Tentando Alpha Vantage para {symbol} com chave terminando em ...{alphavantage_key[-4:] if len(alphavantage_key) > 4 else alphavantage_key}.")

    try:
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": alphavantage_key
        }
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Alpha Vantage: {data['Error Message']}")
        if "Information" in data and "limit" in data["Information"].lower():
            raise ValueError(f"API Alpha Vantage: Limite atingido - {data['Information']}")

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            raise ValueError("Dados 'Time Series (Daily)' não encontrados na Alpha Vantage.")

        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        column_mapping = {
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. adjusted close": "Adj Close", # Usaremos 'Close' como ajustado
            "6. volume": "Volume"
        }
        df = df.rename(columns=column_mapping)
        # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED já retorna OHLC ajustado. 'Close' é o ajustado.
        # Se 'Adj Close' existir, pode ser usado para verificação, mas 'Close' é o que usamos.

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Coluna essencial '{col}' ausente (Alpha Vantage) após renomeação.")

        df[required_columns] = df[required_columns].apply(pd.to_numeric, errors='coerce')
        df = df[required_columns].ffill().dropna() # Ffill antes de dropna

        if df.empty or not all(col in df.columns for col in required_columns) or df[required_columns].isnull().any().any():
            raise ValueError("DataFrame vazio ou com NaNs (Alpha Vantage) após processamento.")

        if _status_placeholder:
            _status_placeholder.text(f"Dados de {symbol} (Alpha Vantage) obtidos.")
        logger.info(f"Dados históricos para {symbol} obtidos com sucesso da Alpha Vantage.")
        return df

    except (requests.exceptions.RequestException, ValueError, KeyError) as e_alpha:
        logger.warning(f"Falha com Alpha Vantage para {symbol}: {e_alpha}. Tentando yFinance.")
        if _status_placeholder:
            _status_placeholder.text(f"Falha Alpha Vantage ({e_alpha}). Tentando yFinance para {symbol}...")
        
        # Fallback para yFinance
        try:
            stock = yf.Ticker(symbol)
            # Usar auto_adjust=True para que Open, High, Low, Close já sejam ajustados
            hist = stock.history(period="max", interval="1d", auto_adjust=True, actions=True)

            if hist.empty:
                raise ValueError("Dados históricos do yFinance estão vazios.")

            # Renomear colunas para garantir consistência, embora auto_adjust=True geralmente já use os nomes corretos.
            hist = hist.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            })
            
            for col in required_columns:
                if col not in hist.columns:
                    raise ValueError(f"Coluna essencial '{col}' ausente nos dados do yFinance.")

            hist = hist[required_columns].ffill().dropna()
            hist.index = pd.to_datetime(hist.index.date) # Normalizar índice para data

            if hist.empty or not all(col in hist.columns for col in required_columns) or hist[required_columns].isnull().any().any():
                raise ValueError("DataFrame vazio ou com NaNs (yFinance) após processamento.")

            if _status_placeholder:
                _status_placeholder.text(f"Dados de {symbol} (yFinance) obtidos.")
            logger.info(f"Dados históricos para {symbol} obtidos com sucesso do yFinance.")
            return hist

        except Exception as e_yf:
            logger.error(f"Falha em ambas as fontes (Alpha Vantage e yFinance) para {symbol}: AV({e_alpha}), YF({e_yf})")
            if _status_placeholder:
                _status_placeholder.text(f"Falha em ambas as fontes para {symbol}: {e_yf}")
            st.error(f"Falha em ambas as fontes para {symbol}: AV: {e_alpha}, YF: {e_yf}")
            return pd.DataFrame(columns=required_columns)


@st.cache_data(ttl=3600)
def get_finnhub_options(symbol: str, _status_placeholder: Optional[st.empty] = None) -> Optional[Dict[str, Any]]:
    """Obter dados de opções da API Finnhub."""
    finnhub_key = get_api_key("finnhub")
    logger.info(f"Tentando obter dados de opções do Finnhub para {symbol}.")
    if finnhub_key == DEFAULT_API_KEYS["finnhub"] and "d0db4g" in finnhub_key: # Heurística para chave demo
        st.info("A chave DEMO do Finnhub tem limitações para dados de opções. Resultados podem ser incompletos.")
        logger.warning("Usando chave DEMO do Finnhub para opções. Dados podem ser limitados.")

    if _status_placeholder:
        _status_placeholder.text(f"Buscando opções (Finnhub) para {symbol}...")
    try:
        base_url = "https://finnhub.io/api/v1/stock/option-chain"
        params = {"symbol": symbol, "token": finnhub_key}
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("s") == "no_data" or not data.get("data"):
            logger.warning(f"Nenhum dado de opção encontrado no Finnhub para {symbol}. Resposta: {data}")
            st.debug(f"Nenhum dado de opção encontrado no Finnhub para {symbol}.")
            return None
        if "error" in data:
            logger.error(f"Erro da API Finnhub ao buscar opções para {symbol}: {data['error']}")
            st.error(f"Erro da API Finnhub: {data['error']}")
            return None
        if _status_placeholder:
            _status_placeholder.text(f"Opções (Finnhub) para {symbol} obtidas.")
        logger.info(f"Dados de opções do Finnhub para {symbol} obtidos com sucesso.")
        return data
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Erro de conexão ao buscar opções do Finnhub para {symbol}: {e_req}")
        st.error(f"Erro de conexão (Finnhub opções) para {symbol}: {e_req}")
        return None
    except Exception as e_gen:
        logger.error(f"Erro inesperado ao obter opções do Finnhub para {symbol}: {e_gen}")
        st.error(f"Erro inesperado (Finnhub opções) para {symbol}: {e_gen}")
        return None

def is_brazilian_ticker(ticker: str) -> bool:
    return bool(re.search(r'\.SA$', ticker, re.IGNORECASE))

def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    # Heurística aprimorada para tickers brasileiros
    if not ticker.endswith(".SA") and not '.' in ticker:
        if re.match(r"^[A-Z]{4}[3456]$", ticker) or \
           re.match(r"^[A-Z]{4}11$", ticker) or \
           re.match(r"^[A-Z]{4}[78]$", ticker): # Ações menos comuns
            logger.debug(f"Normalizando ticker brasileiro: {ticker} -> {ticker}.SA")
            return f"{ticker}.SA"
    return ticker

def yang_zhang(price_data: pd.DataFrame, window: int = 30, trading_periods: int = 252, return_last_only: bool = True) -> Any:
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in price_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        msg = f"Dados de preço incompletos para Yang-Zhang. Colunas ausentes: {missing_cols}. Volatilidade não será calculada."
        logger.warning(msg)
        st.warning(msg)
        return np.nan if return_last_only else pd.Series([np.nan] * len(price_data))

    if price_data.empty or len(price_data) < window + 1:
        msg = f"Dados insuficientes para Yang-Zhang (necessário: {window+1}, disponível: {len(price_data)})."
        logger.warning(msg)
        st.warning(msg)
        return np.nan if return_last_only else pd.Series([np.nan] * len(price_data))

    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])

    safe_open = price_data['Open'].replace(0, np.nan)
    safe_close_shifted = price_data['Close'].shift(1).replace(0, np.nan)
    log_oc = np.log(safe_open / safe_close_shifted)

    log_oc_sq = log_oc ** 2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    window_adj_periods = max(1, window - 1)

    open_var_sum = log_oc_sq.rolling(window=window, min_periods=window).sum()
    rs_var_sum = rs.rolling(window=window, min_periods=window).sum()

    open_var = open_var_sum / window_adj_periods
    rs_var = rs_var_sum / window_adj_periods

    k = 0.34 / (1.34 + (window + 1) / max(1, window - 1))

    volatility_series = np.sqrt(open_var + k * rs_var) * np.sqrt(trading_periods)
    volatility_series = volatility_series.dropna()

    if volatility_series.empty:
        logger.warning("Série de volatilidade Yang-Zhang resultante está vazia.")
        return np.nan
    return volatility_series.iloc[-1] if return_last_only else volatility_series

def build_term_structure(days: List[float], ivs: List[float]) -> Callable[[float], float]:
    days_arr = np.array(days, dtype=float)
    ivs_arr = np.array(ivs, dtype=float)

    valid_indices = ~np.isnan(ivs_arr) & ~np.isnan(days_arr) & (ivs_arr > 1e-5)
    days_arr = days_arr[valid_indices]
    ivs_arr = ivs_arr[valid_indices]

    if len(days_arr) < 1: return lambda x: np.nan
    if len(days_arr) == 1: return lambda x: ivs_arr[0]

    sort_idx = days_arr.argsort()
    days_sorted, ivs_sorted = days_arr[sort_idx], ivs_arr[sort_idx]

    unique_days, unique_idx = np.unique(days_sorted, return_index=True)
    unique_ivs = ivs_sorted[unique_idx]

    if len(unique_days) < 2:
        return lambda x: unique_ivs[0] if len(unique_ivs) > 0 else np.nan

    try:
        spline = interp1d(unique_days, unique_ivs, kind='linear', fill_value="extrapolate")
        return lambda dte_val: float(spline(dte_val)) if pd.notna(spline(dte_val)) else np.nan
    except Exception as e:
        logger.error(f"Erro ao criar spline para estrutura a termo: {e}. Dias: {unique_days}, IVs: {unique_ivs}")
        return lambda x: np.nan


class OptionChain:
    def __init__(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame):
        self.calls = calls_df
        self.puts = puts_df

def get_options_data(ticker_str: str, status_placeholder: Optional[st.empty] = None) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]], Optional[float]]:
    current_price: Optional[float] = None
    logger.info(f"Iniciando obtenção de dados de opções para {ticker_str}.")
    
    if status_placeholder: status_placeholder.text(f"Obtendo preço atual para {ticker_str}...")
    try:
        stock_yf_obj = yf.Ticker(ticker_str) # Criar objeto uma vez
        current_price = get_current_price(stock_yf_obj, status_placeholder)
        if current_price is None:
            current_price = get_current_price_alternative(ticker_str, get_api_key("alphavantage"), status_placeholder)
        if current_price is None:
            logger.error(f"Preço atual não pôde ser obtido para {ticker_str} via yFinance ou AlphaVantage.")
            st.error(f"Preço atual não pôde ser obtido para {ticker_str}.")
            # Não retornar ainda, tentar Finnhub mesmo sem preço, mas muitas calcs falharão
    except Exception as e_price_init:
        logger.error(f"Erro inicial ao obter preço para {ticker_str}: {e_price_init}")
        st.warning(f"Problema ao obter preço inicial para {ticker_str}: {e_price_init}")


    # Tentar yFinance para opções
    if status_placeholder: status_placeholder.text(f"Buscando opções (yFinance) para {ticker_str}...")
    try:
        if 'stock_yf_obj' not in locals(): # Se falhou muito cedo
             stock_yf_obj = yf.Ticker(ticker_str)

        if not stock_yf_obj.options or len(stock_yf_obj.options) == 0:
            raise ValueError("yFinance não retornou datas de expiração.")

        exp_dates_yf = list(stock_yf_obj.options)
        today = datetime.today().date()
        # Permitir opções que expiram hoje (DTE=0) ou no dia anterior (DTE=-1 para liquidação)
        filtered_dates_yf = [date_str for date_str in exp_dates_yf
                             if (datetime.strptime(date_str, "%Y-%m-%d").date() - today).days >= -1]

        if not filtered_dates_yf:
            raise ValueError("yFinance sem datas de expiração futuras, para hoje ou ontem próximo.")

        options_chains_yf = {}
        for date_str in filtered_dates_yf:
            try:
                chain_data = stock_yf_obj.option_chain(date_str)
                if (chain_data.calls is not None and not chain_data.calls.empty) or \
                   (chain_data.puts is not None and not chain_data.puts.empty):
                    options_chains_yf[date_str] = chain_data
                else:
                    logger.debug(f"Cadeia de opções vazia (yFinance) para {ticker_str} em {date_str}.")
            except Exception as e_chain:
                logger.warning(f"Falha ao obter cadeia yFinance para {ticker_str} em {date_str}: {e_chain}")

        if not options_chains_yf:
            raise ValueError("yFinance sem cadeias de opções válidas após filtragem.")

        if status_placeholder: status_placeholder.text(f"Opções (yFinance) para {ticker_str} obtidas.")
        logger.info(f"Dados de opções (yFinance) para {ticker_str} obtidos com sucesso.")
        return filtered_dates_yf, options_chains_yf, current_price

    except Exception as e_yf_options:
        logger.warning(f"Falha com yFinance para opções de {ticker_str} ({e_yf_options}). Tentando Finnhub...")
        if status_placeholder: status_placeholder.text(f"Falha yFinance ({e_yf_options}). Tentando Finnhub...")

        # Fallback para Finnhub
        finnhub_data = get_finnhub_options(ticker_str, _status_placeholder=status_placeholder)

        if not finnhub_data or finnhub_data.get("s") == "no_data" or not finnhub_data.get("data"):
            logger.error(f"Não foi possível obter dados de opções do Finnhub para {ticker_str}.")
            st.error(f"Não foi possível obter dados de opções (Finnhub) para {ticker_str}.")
            return None, None, current_price # Retorna preço se obtido antes

        # Se o preço atual ainda não foi obtido (raro, mas possível)
        if current_price is None:
            logger.info(f"Tentando obter preço atual para {ticker_str} novamente antes de processar Finnhub.")
            current_price = get_current_price_alternative(ticker_str, get_api_key("alphavantage"), status_placeholder)
            if current_price is None and 'stock_yf_obj' in locals():
                current_price = get_current_price(stock_yf_obj, status_placeholder)
            if current_price is None:
                 logger.error(f"Preço atual INDISPONÍVEL para {ticker_str} para processar dados Finnhub.")
                 st.error(f"Preço atual INDISPONÍVEL para {ticker_str}. Dados de opções Finnhub não podem ser processados adequadamente.")
                 return None, None, None


        exp_dates_fh_list: List[str] = []
        options_chains_fh: Dict[str, OptionChain] = {}
        finnhub_option_data_list = finnhub_data.get('data', [])

        if not finnhub_option_data_list:
            logger.error(f"Dados de opções Finnhub ('data' field) vazios para {ticker_str}.")
            st.error(f"Dados de opções Finnhub vazios para {ticker_str}.")
            return None, None, current_price

        processed_exp_dates_fh_set = set()
        today_date_obj = datetime.today().date()

        if isinstance(finnhub_option_data_list, list) and len(finnhub_option_data_list) > 0:
            for expiry_group in finnhub_option_data_list:
                exp_ts = expiry_group.get('expirationDate')
                if not exp_ts:
                    logger.debug(f"Finnhub: 'expirationDate' ausente no grupo: {expiry_group}")
                    continue

                exp_date_obj = datetime.fromtimestamp(exp_ts).date()
                if (exp_date_obj - today_date_obj).days < -1: # Permitir DTE >= -1
                    continue

                exp_str = exp_date_obj.strftime('%Y-%m-%d')
                if exp_str in processed_exp_dates_fh_set:
                    continue

                calls_list_fh, puts_list_fh = [], []
                options_for_expiry = expiry_group.get('options', {})

                if isinstance(options_for_expiry, dict):
                    for contract_type, contracts in options_for_expiry.items(): # CALL, PUT
                        if isinstance(contracts, list):
                            for contract in contracts:
                                if contract and contract.get('strike') is not None:
                                    if contract_type == 'CALL': calls_list_fh.append(contract)
                                    elif contract_type == 'PUT': puts_list_fh.append(contract)
                else:
                    logger.debug(f"Finnhub: campo 'options' para {exp_str} não é um dict: {type(options_for_expiry)}")
                    continue
                
                if not calls_list_fh and not puts_list_fh:
                    logger.debug(f"Finnhub: Sem contratos de call ou put válidos para {exp_str}")
                    continue

                rename_map_fh = {
                    'lastPrice': 'lastPrice', 'impVol': 'impliedVolatility',
                    'vol': 'volume', 'openInterest': 'openInterest',
                    'strike': 'strike', 'bid': 'bid', 'ask': 'ask'
                }
                common_cols = ['strike', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest', 'bid', 'ask']

                calls_df_fh = pd.DataFrame(calls_list_fh).rename(columns=rename_map_fh)
                puts_df_fh = pd.DataFrame(puts_list_fh).rename(columns=rename_map_fh)

                for df_fh in [calls_df_fh, puts_df_fh]:
                    for col in common_cols:
                        if col not in df_fh.columns:
                            df_fh[col] = np.nan # Adicionar colunas ausentes com NaN

                options_chains_fh[exp_str] = OptionChain(calls_df_fh, puts_df_fh)
                processed_exp_dates_fh_set.add(exp_str)

        exp_dates_fh_list = sorted(list(processed_exp_dates_fh_set))

        if not options_chains_fh:
            logger.error(f"Nenhuma cadeia de opções Finnhub válida para {ticker_str} após processamento.")
            st.error(f"Nenhuma cadeia de opções Finnhub válida para {ticker_str} após processamento.")
            return None, None, current_price

        if status_placeholder: status_placeholder.text(f"Opções (Finnhub) para {ticker_str} obtidas.")
        logger.info(f"Dados de opções (Finnhub) para {ticker_str} obtidos com sucesso.")
        return exp_dates_fh_list, options_chains_fh, current_price


def get_current_price(yf_ticker_obj: yf.Ticker, status_placeholder: Optional[st.empty] = None) -> Optional[float]:
    logger.debug(f"Tentando obter preço atual (yFinance) para {yf_ticker_obj.ticker}.")
    try:
        info = yf_ticker_obj.info
        current_price_info = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))
        if current_price_info is not None:
            logger.info(f"Preço atual (yFinance info) para {yf_ticker_obj.ticker}: {current_price_info}")
            return float(current_price_info)
        logger.debug(f"Preço não encontrado em yf_ticker_obj.info para {yf_ticker_obj.ticker}. Tentando histórico.")

        # Fallback para histórico recente se .info falhar
        # Tentar intraday primeiro para mercados abertos
        hist_intraday = yf_ticker_obj.history(period='2d', interval='1m', auto_adjust=True, prepost=True)
        if not hist_intraday.empty and 'Close' in hist_intraday.columns and not hist_intraday['Close'].dropna().empty:
            last_price = hist_intraday['Close'].dropna().iloc[-1]
            logger.info(f"Preço atual (yFinance histórico 1m) para {yf_ticker_obj.ticker}: {last_price}")
            return float(last_price)
        
        logger.debug(f"Histórico intraday (1m) vazio ou sem preço para {yf_ticker_obj.ticker}. Tentando histórico diário.")
        hist_daily = yf_ticker_obj.history(period='5d', interval='1d', auto_adjust=True)
        if not hist_daily.empty and 'Close' in hist_daily.columns and not hist_daily['Close'].dropna().empty:
            last_close = hist_daily['Close'].dropna().iloc[-1]
            logger.info(f"Preço atual (yFinance histórico 1d) para {yf_ticker_obj.ticker}: {last_close}")
            return float(last_close)

        logger.warning(f"yFinance history (1m e 1d) vazia ou sem preço para {yf_ticker_obj.ticker}")
        return None

    except Exception as e:
        logger.warning(f"Erro ao buscar preço (yFinance) para {yf_ticker_obj.ticker}: {e}")
        if status_placeholder: status_placeholder.text(f"Aviso yFinance preço: {e}")
        return None

def get_current_price_alternative(symbol: str, api_key: str, status_placeholder: Optional[st.empty] = None) -> Optional[float]:
    logger.debug(f"Tentando obter preço atual (AlphaVantage Global Quote) para {symbol}.")
    if status_placeholder: status_placeholder.text(f"Buscando preço (AlphaVantage) para {symbol}...")
    try:
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Global Quote" not in data or not data["Global Quote"]:
            logger.warning(f"Resposta 'Global Quote' vazia/ausente da AlphaVantage para {symbol}. Data: {data}")
            return None

        price_str = data.get("Global Quote", {}).get("05. price")
        if price_str:
            try:
                price = float(price_str)
                logger.info(f"Preço atual (AlphaVantage) para {symbol}: {price}")
                return price
            except ValueError:
                logger.warning(f"Preço da AlphaVantage não é número válido para {symbol}: {price_str}")
                return None
        
        logger.warning(f"Preço não encontrado (AlphaVantage Global Quote) para {symbol}. Resposta: {data}")
        return None
    except requests.exceptions.RequestException as e_req:
        logger.error(f"Erro de conexão em get_current_price_alternative (AlphaVantage) para {symbol}: {e_req}")
        return None
    except Exception as e:
        logger.error(f"Erro geral em get_current_price_alternative (AlphaVantage) para {symbol}: {e}")
        return None

@st.cache_data(ttl=86400) # Cache por um dia
def get_company_profile_finnhub(symbol: str) -> Optional[Dict[str, Any]]:
    """Get company profile from Finnhub, including logo."""
    api_key = get_api_key("finnhub")
    logger.info(f"Buscando perfil da empresa (Finnhub) para {symbol}.")
    if api_key == DEFAULT_API_KEYS["finnhub"] and "d0db4g" in api_key:
        msg = "Finnhub API key (non-demo) é necessária para perfil da empresa. Usando chave DEMO."
        logger.warning(msg)
        st.info(msg)
        # Pode prosseguir com a chave demo, mas pode não retornar tudo
    
    if not api_key:
        logger.warning(f"Chave API Finnhub não configurada, não é possível buscar perfil para {symbol}.")
        return None
        
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        profile = r.json()
        if profile and isinstance(profile, dict) and profile.get('ticker') == symbol: # Check if data is for the requested ticker
            logger.info(f"Perfil Finnhub para {symbol} obtido com sucesso.")
            return profile
        logger.warning(f"Perfil Finnhub para {symbol} não encontrado ou resposta inesperada: {profile}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Não foi possível buscar perfil da empresa no Finnhub para {symbol}: {e}")
        return None
    except Exception as e_ex: # Captura JSONDecodeError e outros
        logger.error(f"Erro inesperado ao buscar/processar perfil Finnhub para {symbol}: {e_ex}")
        return None


@st.cache_data(ttl=3600)
def get_fundamental_data_yf(symbol: str) -> Dict[str, Any]:
    """Get key fundamental data and calendar events from yFinance."""
    logger.info(f"Buscando dados fundamentalistas (yFinance) para {symbol}.")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals: Dict[str, Any] = {}

        if not info:
            logger.warning(f"yFinance .info está vazio para {symbol}.")
            return fundamentals

        fundamentals['name'] = info.get('longName', info.get('shortName', symbol))
        fundamentals['sector'] = info.get('sector', 'N/A')
        fundamentals['industry'] = info.get('industry', 'N/A')
        fundamentals['country'] = info.get('country', 'N/A')
        fundamentals['website'] = info.get('website', 'N/A')
        fundamentals['logo_url'] = info.get('logo_url', None)
        fundamentals['longBusinessSummary'] = info.get('longBusinessSummary', 'N/A')

        keys_to_extract = [
            'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 'pegRatio',
            'priceToSalesTrailing12Months', 'priceToBook', 'enterpriseToRevenue',
            'enterpriseToEbitda', 'profitMargins', 'grossMargins', 'ebitdaMargins',
            'operatingMargins', 'revenueGrowth', 'earningsQuarterlyGrowth',
            'returnOnEquity', 'returnOnAssets', 'debtToEquity', 'currentRatio',
            'quickRatio', 'dividendRate', 'dividendYield', 'beta', '52WeekChange',
            'averageVolume', 'regularMarketPreviousClose', 'regularMarketOpen',
            'regularMarketDayHigh', 'regularMarketDayLow', 'bid', 'ask', 'bidSize', 'askSize'
        ]
        for key in keys_to_extract:
            fundamentals[key] = info.get(key)
        
        # Datas
        ex_div_timestamp = info.get('exDividendDate')
        if ex_div_timestamp and pd.notna(ex_div_timestamp):
            # yfinance pode retornar timestamp em segundos
            fundamentals['exDividendDate'] = pd.to_datetime(ex_div_timestamp, unit='s', errors='coerce').strftime('%Y-%m-%d')
        else:
            fundamentals['exDividendDate'] = "N/A"

        # Calendário de Eventos
        try:
            calendar_events_df = stock.calendar
            if calendar_events_df is not None and not calendar_events_df.empty:
                earnings_data = calendar_events_df.get('Earnings Date', pd.Series(dtype='object')) # Retorna Series
                if earnings_data is not None and not earnings_data.empty:
                    # yf retorna datas como Timestamps ou str. Formatamos para str.
                    fundamentals['earningsDates'] = [
                        d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d)
                        for d in earnings_data if pd.notna(d)
                    ]
                else:
                    fundamentals['earningsDates'] = ["N/A"]
                
                # Tentar obter Ex-Dividend Date do calendário também, se ausente ou "N/A" em info
                if fundamentals.get('exDividendDate') == "N/A":
                    ex_div_cal_series = calendar_events_df.get('Ex-Dividend Date', pd.Series(dtype='object'))
                    if ex_div_cal_series is not None and not ex_div_cal_series.empty:
                        ex_div_cal_series = pd.to_datetime(ex_div_cal_series, errors='coerce').dropna()
                        if not ex_div_cal_series.empty:
                            today = pd.Timestamp(datetime.now().date())
                            future_ex_div = ex_div_cal_series[ex_div_cal_series >= today]
                            past_ex_div = ex_div_cal_series[ex_div_cal_series < today]
                            calendar_ex_div_date_str = None
                            if not future_ex_div.empty: calendar_ex_div_date_str = future_ex_div.min().strftime('%Y-%m-%d')
                            elif not past_ex_div.empty: calendar_ex_div_date_str = past_ex_div.max().strftime('%Y-%m-%d') # Mais recente passado
                            if calendar_ex_div_date_str:
                                fundamentals['exDividendDate'] = calendar_ex_div_date_str
            else:
                fundamentals['earningsDates'] = ["N/A"]
        except Exception as e_cal:
            logger.warning(f"Erro ao processar calendário de eventos yFinance para {symbol}: {e_cal}")
            fundamentals['earningsDates'] = ["N/A"]


        fundamentals['recordDate'] = "N/A (difícil de obter via API gratuita)" # Esta informação é raramente disponível

        # Histórico de Dividendos
        dividends_history = stock.dividends
        if dividends_history is not None and not dividends_history.empty:
            fundamentals['recent_dividends'] = dividends_history.tail().reset_index().rename(
                columns={'Date': 'Data', 'Dividends': 'Valor'}
            ).to_dict('records')
            for record in fundamentals['recent_dividends']:
                if isinstance(record.get('Data'), pd.Timestamp):
                    record['Data'] = record['Data'].strftime('%Y-%m-%d')
        else:
            fundamentals['recent_dividends'] = []
        
        logger.info(f"Dados fundamentalistas (yFinance) para {symbol} obtidos.")
        return fundamentals

    except Exception as e:
        logger.error(f"Erro ao buscar dados fundamentalistas (yFinance) para {symbol}: {e}")
        st.warning(f"Erro ao buscar dados fundamentalistas (yFinance) para {symbol}: {e}")
        return {}

# --- Dicionário de Empresas (Screener) ---
TOP_US_COMPANIES = {
    "Apple Inc.": "AAPL", "Microsoft Corp.": "MSFT", "Alphabet Inc. (Google)": "GOOGL",
    "Amazon.com Inc.": "AMZN", "NVIDIA Corporation": "NVDA", "Meta Platforms, Inc.": "META",
    "Tesla, Inc.": "TSLA", "Berkshire Hathaway Inc.": "BRK-B", "Eli Lilly and Company": "LLY",
    "JPMorgan Chase & Co.": "JPM" # Lista abreviada para exemplo
    # Adicione mais conforme necessário
}

# --- Lógica Principal de Análise ---
def compute_recommendation(ticker_input: str, status_placeholder: st.empty, progress_bar: st.progress) -> Union[Dict[str, Any], str]:
    try:
        logger.info(f"Iniciando compute_recommendation para ticker_input: {ticker_input}")
        normalized_ticker = normalize_ticker(ticker_input)
        logger.info(f"Ticker normalizado: {normalized_ticker}")
        
        status_placeholder.text(f"Buscando dados históricos para {normalized_ticker}...")
        price_history = get_stock_data(normalized_ticker, _status_placeholder=status_placeholder)
        progress_bar.progress(25)

        if price_history.empty or not all(col in price_history.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']) \
           or price_history[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
            msg = f"Não foi possível obter dados históricos (OHLCV) completos e válidos para {normalized_ticker}."
            logger.error(msg)
            st.error(msg)
            return f"Erro: {msg}"
            
        exp_dates, options_chains, underlying_price = get_options_data(normalized_ticker, status_placeholder=status_placeholder)
        progress_bar.progress(50) 
        
        if underlying_price is None or pd.isna(underlying_price):
            logger.warning(f"Preço base do ativo ({normalized_ticker}) não pôde ser determinado inicialmente.")
            if not price_history.empty and 'Close' in price_history.columns and not price_history['Close'].empty:
                underlying_price = price_history['Close'].iloc[-1]
                msg_info = f"Usando último preço de fechamento histórico ({underlying_price:.2f}) como preço base para {normalized_ticker}."
                logger.info(msg_info)
                st.info(msg_info)
                if underlying_price is None or pd.isna(underlying_price): 
                     err_msg = f"Preço base do ativo ({normalized_ticker}) não pôde ser determinado de nenhuma fonte."
                     logger.error(err_msg)
                     return f"Erro: {err_msg}"
            else:
                err_msg = f"Preço base do ativo ({normalized_ticker}) não pôde ser determinado e histórico de preços indisponível."
                logger.error(err_msg)
                return f"Erro: {err_msg}"
        logger.info(f"Preço base para {normalized_ticker}: {underlying_price:.2f}")

        atm_iv: Dict[str, float] = {}
        straddle_info: Optional[Dict[str, Any]] = None
        dtes: List[float] = []
        ivs: List[float] = [] 

        if not exp_dates or not options_chains: 
            msg_warn = f"Não há dados de opções para {normalized_ticker}. Métricas de VI podem não estar disponíveis."
            logger.warning(msg_warn)
            st.warning(msg_warn)
        else:
            status_placeholder.text(f"Calculando métricas de volatilidade para {normalized_ticker}...")
            
            for exp_date_str in exp_dates:
                chain = options_chains.get(exp_date_str)
                if not chain: 
                    logger.debug(f"Nenhuma cadeia de opções para {exp_date_str} em {normalized_ticker}.")
                    continue

                calls_df = getattr(chain, 'calls', pd.DataFrame())
                puts_df = getattr(chain, 'puts', pd.DataFrame())
                
                # Limpeza e conversão de dados de opções
                for df_opt in [calls_df, puts_df]:
                    if not df_opt.empty:
                        for col_to_numeric in ['strike', 'impliedVolatility', 'bid', 'ask', 'lastPrice']:
                            if col_to_numeric in df_opt.columns:
                                df_opt[col_to_numeric] = pd.to_numeric(df_opt[col_to_numeric], errors='coerce')
                
                valid_calls = calls_df[(calls_df['impliedVolatility'].notna()) & (calls_df['impliedVolatility'] > 1e-4) & (calls_df['strike'].notna())].copy() if not calls_df.empty else pd.DataFrame()
                valid_puts = puts_df[(puts_df['impliedVolatility'].notna()) & (puts_df['impliedVolatility'] > 1e-4) & (puts_df['strike'].notna())].copy() if not puts_df.empty else pd.DataFrame()

                if valid_calls.empty or valid_puts.empty: 
                    logger.debug(f"Sem calls/puts válidos com IV e Strike para {exp_date_str} ({normalized_ticker}) após filtragem.")
                    continue
                
                valid_calls['diff_abs'] = (valid_calls['strike'] - underlying_price).abs()
                valid_puts['diff_abs'] = (valid_puts['strike'] - underlying_price).abs()
                
                if valid_calls['diff_abs'].empty or valid_puts['diff_abs'].empty : continue 

                atm_call_row = valid_calls.loc[valid_calls['diff_abs'].idxmin()] if not valid_calls.empty else pd.Series(dtype='float64')
                atm_put_row = valid_puts.loc[valid_puts['diff_abs'].idxmin()] if not valid_puts.empty else pd.Series(dtype='float64')
                
                if atm_call_row.empty or atm_put_row.empty: continue

                current_atm_iv_val = (atm_call_row.get('impliedVolatility', np.nan) + atm_put_row.get('impliedVolatility', np.nan)) / 2.0
                if pd.notna(current_atm_iv_val) and current_atm_iv_val > 0:
                     atm_iv[exp_date_str] = current_atm_iv_val
                
                # Calcular exemplo de Straddle (pega o primeiro ATM viável)
                if straddle_info is None:
                    c_bid, c_ask, c_last = atm_call_row.get('bid'), atm_call_row.get('ask'), atm_call_row.get('lastPrice')
                    p_bid, p_ask, p_last = atm_put_row.get('bid'), atm_put_row.get('ask'), atm_put_row.get('lastPrice')

                    c_mid_price = np.nan
                    if pd.notna(c_bid) and pd.notna(c_ask) and c_bid > 0 and c_ask > 0 and c_ask >= c_bid: c_mid_price = (c_bid + c_ask) / 2.0
                    elif pd.notna(c_last) and c_last > 0: c_mid_price = c_last

                    p_mid_price = np.nan
                    if pd.notna(p_bid) and pd.notna(p_ask) and p_bid > 0 and p_ask > 0 and p_ask >= p_bid: p_mid_price = (p_bid + p_ask) / 2.0
                    elif pd.notna(p_last) and p_last > 0: p_mid_price = p_last
                    
                    if pd.notna(c_mid_price) and c_mid_price > 0 and pd.notna(p_mid_price) and p_mid_price > 0:
                        try:
                            exp_obj_dt = datetime.strptime(exp_date_str.split('T')[0], "%Y-%m-%d").date()
                            dte_val_days = (exp_obj_dt - datetime.today().date()).days
                            if dte_val_days >= 0: # Apenas straddles com DTE >= 0 para o exemplo
                                total_cost_straddle = c_mid_price + p_mid_price
                                straddle_info = {
                                    'expiry': exp_obj_dt.strftime("%Y-%m-%d"), 'days_to_expiry': dte_val_days,
                                    'call_strike': atm_call_row['strike'], 'put_strike': atm_put_row['strike'],
                                    'call_price': c_mid_price, 'put_price': p_mid_price, 'total_cost': total_cost_straddle,
                                    'expected_move_pct': (total_cost_straddle / underlying_price) * 100 if underlying_price > 0 else 0
                                }
                                logger.info(f"Exemplo Straddle ATM calculado para {normalized_ticker}, exp {exp_date_str}: {straddle_info}")
                        except Exception as e_straddle:
                            logger.warning(f"Erro ao calcular straddle para {exp_date_str} ({normalized_ticker}): {e_straddle}")
            
            progress_bar.progress(70)
            status_placeholder.text(f"Calculando superfície de volatilidade para {normalized_ticker}...")
            
            if atm_iv: 
                today_date_calc = datetime.today().date()
                for exp_str_loop, iv_val_loop in sorted(atm_iv.items()): 
                    try:
                        exp_date_obj_loop = datetime.strptime(exp_str_loop.split('T')[0], "%Y-%m-%d").date()
                        days_to_exp_val = (exp_date_obj_loop - today_date_calc).days
                        if days_to_exp_val >= -1: # Permitir DTE = -1, 0, ...
                            dtes.append(days_to_exp_val)
                            ivs.append(iv_val_loop)
                    except ValueError as e_date_parse: 
                        logger.debug(f"Erro ao parsear data de expiração '{exp_str_loop}' para {normalized_ticker}: {e_date_parse}")
            else: 
                 logger.warning(f"Não há dados de IV ATM suficientes para {normalized_ticker} para construir a estrutura a termo.")
        
        iv30: float = np.nan
        ts_slope_0_45: float = np.nan

        if dtes and ivs and len(dtes) >=1 : 
            term_spline_func = build_term_structure(dtes, ivs)
            iv30 = term_spline_func(30) 
            if pd.isna(iv30) or iv30 <=0: iv30 = np.nan
            logger.info(f"IV30 para {normalized_ticker}: {iv30}")

            s_dtes_for_slope = sorted(list(set(d for d in dtes if d >=0))) # Apenas DTEs >= 0 para inclinação
            if len(s_dtes_for_slope) >= 2:
                d_start_slope = 0 if 0 in s_dtes_for_slope else (s_dtes_for_slope[0] if s_dtes_for_slope else np.nan)
                d_end_slope = 45
                
                val_at_start_slope = term_spline_func(d_start_slope) if pd.notna(d_start_slope) else np.nan
                val_at_end_slope = term_spline_func(d_end_slope)

                if pd.notna(val_at_start_slope) and pd.notna(val_at_end_slope) and pd.notna(d_start_slope) and (d_end_slope - d_start_slope) > 1e-6 :
                    ts_slope_0_45 = (val_at_end_slope - val_at_start_slope) / (d_end_slope - d_start_slope)
                else: # Fallback para os dois primeiros pontos se o range específico falhar
                    d1_slope, d2_slope = s_dtes_for_slope[0], s_dtes_for_slope[1]
                    val_at_d1_slope = term_spline_func(d1_slope)
                    val_at_d2_slope = term_spline_func(d2_slope)
                    if pd.notna(val_at_d1_slope) and pd.notna(val_at_d2_slope) and (d2_slope - d1_slope) > 1e-6:
                         ts_slope_0_45 = (val_at_d2_slope - val_at_d1_slope) / (d2_slope - d1_slope)
                    else: logger.debug(f"Não foi possível calcular a inclinação da estrutura a termo para {normalized_ticker} (pontos insuficientes/inválidos).")
            else: 
                logger.debug(f"Menos de 2 DTEs únicos >=0 para calcular a inclinação para {normalized_ticker}.")
            logger.info(f"Inclinação Estrutura a Termo (0-45d) para {normalized_ticker}: {ts_slope_0_45}")
        else: 
            logger.warning(f"Dados DTE/IV insuficientes para calcular IV30 ou inclinação para {normalized_ticker}.")

        progress_bar.progress(85)
        status_placeholder.text(f"Finalizando análise para {normalized_ticker}...")
        
        yz_val: float = np.nan
        iv30_rv_yang: float = np.nan 

        try:
            yz_val_30 = yang_zhang(price_history, window=30, return_last_only=True)
            if pd.notna(yz_val_30):
                yz_val = yz_val_30
            elif len(price_history) >= 21: 
                yz_val_20 = yang_zhang(price_history, window=20, return_last_only=True)
                if pd.notna(yz_val_20):
                    yz_val = yz_val_20
                    st.info(f"Usando Vol. Histórica (YZ) com janela de 20 dias para {normalized_ticker}.")
            
            if pd.isna(yz_val):
                logger.warning(f"Vol. Histórica (YZ) não pôde ser calculada para {normalized_ticker}.")
            logger.info(f"Volatilidade Histórica YZ para {normalized_ticker}: {yz_val}")
            
            if pd.notna(iv30) and pd.notna(yz_val) and yz_val > 1e-5: # Evitar divisão por zero
                iv30_rv_yang = iv30 / yz_val
            else:
                iv30_rv_yang = np.nan
            logger.info(f"Relação IV30/RV_YZ para {normalized_ticker}: {iv30_rv_yang}")

        except Exception as e_vol_calc: 
            logger.error(f"Erro ao calcular métricas de volatilidade (YZ ou IV/RV) para {normalized_ticker}: {e_vol_calc}")
        
        avg_vol: float = np.nan
        if 'Volume' in price_history and not price_history['Volume'].empty:
            avg_vol = price_history['Volume'].rolling(window=30, min_periods=1).mean().iloc[-1]
            if pd.isna(avg_vol): avg_vol = 0.0
        else: avg_vol = 0.0
        logger.info(f"Volume médio (30d) para {normalized_ticker}: {avg_vol}")

        # Lógica de Recomendação
        rec_text = "Nenhuma estratégia clara recomendada devido a dados insuficientes ou condições não atendidas."
        vol_thresh = 500000 if is_brazilian_ticker(normalized_ticker) else 1500000
        iv_rv_thresh_sell = 1.20
        iv_rv_thresh_buy = 0.85 # Distinto do sell
        ts_slope_contango_strong = 0.001 
        ts_slope_backwardation_strong = -0.001
        
        can_recommend = pd.notna(iv30_rv_yang) and pd.notna(iv30) and pd.notna(yz_val) and pd.notna(ts_slope_0_45) and avg_vol > 0

        if can_recommend:
            slope_desc = "Flat"
            if ts_slope_0_45 >= ts_slope_contango_strong: slope_desc = "Contango Forte"
            elif ts_slope_0_45 <= ts_slope_backwardation_strong: slope_desc = "Backwardation Forte"
            elif ts_slope_0_45 > 0: slope_desc = "Contango Leve"
            elif ts_slope_0_45 < 0: slope_desc = "Backwardation Leve"

            if iv30_rv_yang > iv_rv_thresh_sell and avg_vol > vol_thresh:
                rec_text = f"Venda de Volatilidade (Estrutura: {slope_desc}). IV ({iv30:.2%}) > RV ({yz_val:.2%}). Relação IV/RV: {iv30_rv_yang:.2f}."
            elif iv30_rv_yang < iv_rv_thresh_buy and avg_vol > vol_thresh:
                 rec_text = f"Compra de Volatilidade (Estrutura: {slope_desc}). IV ({iv30:.2%}) < RV ({yz_val:.2%}). Relação IV/RV: {iv30_rv_yang:.2f}."
            else:
                rec_text = "Mercado de volatilidade neutro ou liquidez/condições insuficientes para recomendação."
        logger.info(f"Recomendação para {normalized_ticker}: {rec_text}")
        
        expected_move_str = "N/A"
        if straddle_info and straddle_info.get('total_cost') and underlying_price > 0: # underlying_price deve ser > 0
            expected_move_str = f"{straddle_info['expected_move_pct']:.2f}% (custo: {straddle_info['total_cost']:.2f}, DTE: {straddle_info['days_to_expiry']})"
        
        progress_bar.progress(100)
        if status_placeholder: status_placeholder.empty()
        
        result_dict = {
            'ticker': normalized_ticker, 'underlying_price': underlying_price, 
            'historical_volatility_yz': yz_val,
            'iv30': iv30, 'iv_rv_ratio': iv30_rv_yang, 'avg_volume': avg_vol, 
            'term_structure_slope_0_45': ts_slope_0_45, 'recommendation': rec_text, 
            'price_history_df': price_history, 
            'atm_iv_data': atm_iv if atm_iv else None, 
            'dtes_for_plot': dtes if dtes else [], 
            'ivs_for_plot': ivs if ivs else [], 
            'straddle_example': straddle_info, 
            'expected_move_straddle': expected_move_str
        }
        logger.info(f"Análise para {normalized_ticker} concluída com sucesso. Resultado: { {k:v for k,v in result_dict.items() if k != 'price_history_df'} }") # Log sem o df
        return result_dict

    except Exception as e_main_compute: 
        logger.critical(f"Erro CRÍTICO na função compute_recommendation para {ticker_input}: {e_main_compute}", exc_info=True)
        if progress_bar: progress_bar.progress(100) # Finaliza a barra
        if status_placeholder: status_placeholder.empty()
        return f"Erro crítico na análise para {ticker_input}: {e_main_compute}"


# --- Funções de Plotagem e UI Auxiliares ---
def plot_price_history(df: Optional[pd.DataFrame], ticker: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or 'Close' not in df.columns or len(df) < 2:
        msg = f"Dados de histórico insuficientes para plotar para {ticker}."
        logger.warning(msg)
        st.warning(msg)
        fig.update_layout(title=f"Histórico de Preços para {ticker} - Dados Indisponíveis",
                          xaxis_title='Data', yaxis_title='Preço de Fechamento')
        return fig
    
    # O índice já deve ser DatetimeIndex
    fig = px.line(df.reset_index(), x=df.index.name if df.index.name else 'index', y='Close', 
                  title=f"Histórico de Preços (Ajustado) para {ticker}")
    fig.update_layout(xaxis_title='Data', yaxis_title='Preço de Fechamento')
    return fig

def plot_volatility_surface(dtes_plot: List[float], ivs_plot: List[float], ticker: str, atm_iv_data_plot: Optional[Dict[str, float]]=None) -> go.Figure:
    plot_dtes_final, plot_ivs_final = [], []

    # Priorizar dtes_plot/ivs_plot passados diretamente se válidos
    if dtes_plot and ivs_plot and len(dtes_plot) == len(ivs_plot):
        valid_pairs = [(d, i) for d, i in zip(dtes_plot, ivs_plot) if pd.notna(d) and pd.notna(i) and i > 1e-5 and d >= -1]
        if valid_pairs:
            valid_pairs.sort(key=lambda x: x[0]) # Ordenar por DTE
            plot_dtes_final = [p[0] for p in valid_pairs]
            plot_ivs_final = [p[1] for p in valid_pairs]

    # Fallback para atm_iv_data_plot se dtes_plot/ivs_plot não forem suficientes
    if not plot_dtes_final and atm_iv_data_plot and isinstance(atm_iv_data_plot, dict):
        temp_dtes, temp_ivs = [], []
        today = datetime.today().date()
        for date_str, iv_val_entry in sorted(atm_iv_data_plot.items()): # Ordenar por data de expiração
            try:
                exp_date_obj = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d").date()
                days_to_exp_entry = (exp_date_obj - today).days
                if days_to_exp_entry >= -1 and pd.notna(iv_val_entry) and iv_val_entry > 1e-5:
                    temp_dtes.append(days_to_exp_entry)
                    temp_ivs.append(iv_val_entry)
            except ValueError: continue 
        
        if temp_dtes and temp_ivs: # Já estarão ordenados por DTE devido ao sorted(items) e filtro DTE
             plot_dtes_final, plot_ivs_final = temp_dtes, temp_ivs
    
    fig = go.Figure()
    if not plot_dtes_final or not plot_ivs_final or len(plot_dtes_final) == 0:
        msg = f"Dados de IV ATM insuficientes ou inconsistentes para plotar estrutura a termo para {ticker}."
        logger.warning(msg)
        st.warning(msg)
        fig.update_layout(title=f"Estrutura a Termo da VI ATM para {ticker} - Dados Indisponíveis",
                          xaxis_title="DTE (Dias Para Expirar)", yaxis_title="VI ATM (%)", yaxis_tickformat='.2%')
        return fig

    fig.add_trace(go.Scatter(x=plot_dtes_final, y=plot_ivs_final, mode='lines+markers', name='IV ATM'))
    fig.update_layout(title=f"Estrutura a Termo da VI ATM para {ticker}", 
                      xaxis_title="DTE (Dias Para Expirar)", 
                      yaxis_title="VI ATM (%)", yaxis_tickformat='.2%')
    return fig

@st.cache_data 
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode('utf-8')

# --- Função Principal Streamlit ---
def main():
    st.title("Analisador de Estratégias de Volatilidade para Mercados BR e US")
    
    st.sidebar.header("Configurações")

    # Inputs para API Keys do Usuário
    st.sidebar.subheader("Chaves de API (Opcional)")
    st.session_state.user_alphavantage_api_key = st.sidebar.text_input(
        "Sua Chave Alpha Vantage:", 
        value=st.session_state.get("user_alphavantage_api_key", ""), type="password"
    )
    st.session_state.user_finnhub_api_key = st.sidebar.text_input(
        "Sua Chave Finnhub:", 
        value=st.session_state.get("user_finnhub_api_key", ""), type="password"
    )
    if st.sidebar.button("Limpar Chaves Salvas"):
        st.session_state.user_alphavantage_api_key = ""
        st.session_state.user_finnhub_api_key = ""
        st.sidebar.success("Chaves limpas da sessão.")


    st.sidebar.subheader("Screener (Top US Companies)")
    selected_company_name = st.sidebar.selectbox(
        "Escolha uma empresa:",
        options=["Digite um ticker abaixo..."] + list(TOP_US_COMPANIES.keys()),
        index=0,
        key="screener_selectbox"
    )

    default_ticker = "PETR4" 
    if selected_company_name != "Digite um ticker abaixo...":
        default_ticker = TOP_US_COMPANIES[selected_company_name]

    ticker_input_key = "ticker_input_main" 
    if selected_company_name != "Digite um ticker abaixo..." or ticker_input_key not in st.session_state:
        st.session_state[ticker_input_key] = default_ticker
    elif not st.session_state.get(ticker_input_key) and selected_company_name != "Digite um ticker abaixo...":
         st.session_state[ticker_input_key] = default_ticker


    ticker_input_value = st.sidebar.text_input(
        "Ou digite o ticker do ativo (ex: PETR4, AAPL):", 
        key=ticker_input_key 
    )
    ticker_input_processed = ticker_input_value.strip().upper() if ticker_input_value else ""
    
    status_placeholder = st.sidebar.empty()
    progress_bar_container = st.sidebar.empty() 

    tab1, tab2, tab_company_info, tab_about = st.tabs([
        "📊 Análise de Volatilidade", 
        "💹 Dados Históricos", 
        "🏢 Informações da Empresa",
        "ℹ️ Sobre & Log"
    ])    

    if st.sidebar.button("Analisar Ativo", key="analyze_button"):
        if not ticker_input_processed: 
            st.error("Por favor, insira um ticker.")
            logger.warning("Tentativa de análise com ticker vazio.")
            return
        
        logger.info(f"Botão 'Analisar Ativo' clicado para: {ticker_input_processed}")
        
        current_status_placeholder = status_placeholder 
        if hasattr(current_status_placeholder, 'text'):
            current_status_placeholder.text(f"Iniciando análise para {ticker_input_processed}...")
        
        analysis_progress_bar = None 
        with progress_bar_container: # Cria um container para a barra de progresso
            analysis_progress_bar = st.progress(0)

        normalized_ticker_for_analysis = normalize_ticker(ticker_input_processed)

        # Obter dados da empresa ANTES da análise de volatilidade para preencher a aba
        company_profile_fh = None
        if not is_brazilian_ticker(normalized_ticker_for_analysis): # Finnhub tem melhor cobertura para US
             company_profile_fh = get_company_profile_finnhub(normalized_ticker_for_analysis)
        
        fundamental_data_yf = get_fundamental_data_yf(normalized_ticker_for_analysis)
        if analysis_progress_bar: analysis_progress_bar.progress(10)


        with st.spinner(f'Analisando volatilidade para {normalized_ticker_for_analysis}, por favor aguarde...'):
            # Passar o objeto de barra de progresso real
            analysis_result = compute_recommendation(normalized_ticker_for_analysis, current_status_placeholder, analysis_progress_bar)


        # --- Preencher Aba de Informações da Empresa ---
        with tab_company_info:
            st.header(f"Informações da Empresa: {normalized_ticker_for_analysis}")

            logo_url_to_display = None
            company_name_display = normalized_ticker_for_analysis # Default

            # Priorizar Finnhub para logo e nome se disponível (especialmente US)
            if company_profile_fh and company_profile_fh.get('logo'):
                logo_url_to_display = company_profile_fh.get('logo')
            elif fundamental_data_yf and fundamental_data_yf.get('logo_url'): # Fallback para yfinance
                logo_url_to_display = fundamental_data_yf.get('logo_url')
            
            if fundamental_data_yf and fundamental_data_yf.get('name') not in [None, 'N/A', '']:
                 company_name_display = fundamental_data_yf.get('name')
            elif company_profile_fh and company_profile_fh.get('name') not in [None, 'N/A', '']:
                 company_name_display = company_profile_fh.get('name')


            col_logo, col_name_site = st.columns([1,3])
            with col_logo:
                if logo_url_to_display:
                    st.image(logo_url_to_display, width=100)
                else:
                    st.caption("Logo não disponível")
            with col_name_site:
                st.subheader(company_name_display)
                website_url = None
                if company_profile_fh and company_profile_fh.get('weburl'):
                     website_url = company_profile_fh.get('weburl')
                elif fundamental_data_yf and fundamental_data_yf.get('website'):
                     website_url = fundamental_data_yf.get('website')
                
                if website_url:
                     if not website_url.startswith(('http://', 'https://')):
                         website_url = 'https://' + website_url # Assegurar esquema
                     st.markdown(f"🌐 [{website_url.replace('https://','').replace('http://','')}]({website_url})", unsafe_allow_html=True)
            
            st.subheader("Sumário do Negócio")
            summary = "N/A"
            if fundamental_data_yf and fundamental_data_yf.get('longBusinessSummary') not in [None, 'N/A','']:
                summary = fundamental_data_yf['longBusinessSummary']
            elif company_profile_fh and company_profile_fh.get('description') not in [None, 'N/A','']: # Finnhub usa 'description'
                summary = company_profile_fh['description']
            st.write(summary)

            st.subheader("Dados Fundamentalistas Chave")
            if fundamental_data_yf or company_profile_fh:
                metrics_col1, metrics_col2 = st.columns(2)
                def format_large_number(num: Any) -> str:
                    if pd.isna(num) or num is None: return "N/A"
                    try:
                        num_float = float(num)
                        if abs(num_float) >= 1e12: return f"{num_float/1e12:.2f} T"
                        if abs(num_float) >= 1e9: return f"{num_float/1e9:.2f} B"
                        if abs(num_float) >= 1e6: return f"{num_float/1e6:.2f} M"
                        return f"{num_float:,.2f}" # Milhares e menores com vírgula
                    except (ValueError, TypeError): return "N/A"

                with metrics_col1:
                    sector_val = fundamental_data_yf.get('sector', company_profile_fh.get('finnhubIndustry', "N/A") if company_profile_fh else "N/A")
                    st.metric("Setor", sector_val)
                    country_val = fundamental_data_yf.get('country', company_profile_fh.get('country', "N/A") if company_profile_fh else "N/A")
                    st.metric("País", country_val)
                    
                    market_cap_yf = fundamental_data_yf.get('marketCap')
                    market_cap_fh = (company_profile_fh.get('marketCapitalization', np.nan) * 1e6) if company_profile_fh and company_profile_fh.get('marketCapitalization') is not None else np.nan
                    market_cap_display = market_cap_yf if pd.notna(market_cap_yf) else market_cap_fh
                    st.metric("Market Cap", format_large_number(market_cap_display))
                    
                    st.metric("P/E (Trailing)", f"{fundamental_data_yf.get('trailingPE'):.2f}" if pd.notna(fundamental_data_yf.get('trailingPE')) else "N/A")
                    st.metric("P/S (Trailing)", f"{fundamental_data_yf.get('priceToSalesTrailing12Months'):.2f}" if pd.notna(fundamental_data_yf.get('priceToSalesTrailing12Months')) else "N/A")
                    st.metric("Dividend Yield", f"{fundamental_data_yf.get('dividendYield', 0)*100:.2f}%" if pd.notna(fundamental_data_yf.get('dividendYield')) else "N/A")

                with metrics_col2:
                    industry_val = fundamental_data_yf.get('industry', "N/A") # yf é geralmente bom para isto
                    st.metric("Indústria", industry_val)
                    st.metric("Beta", f"{fundamental_data_yf.get('beta'):.2f}" if pd.notna(fundamental_data_yf.get('beta')) else "N/A")
                    st.metric("Enterprise Value", format_large_number(fundamental_data_yf.get('enterpriseValue')))
                    st.metric("P/E (Forward)", f"{fundamental_data_yf.get('forwardPE'):.2f}" if pd.notna(fundamental_data_yf.get('forwardPE')) else "N/A")
                    st.metric("P/B", f"{fundamental_data_yf.get('priceToBook'):.2f}" if pd.notna(fundamental_data_yf.get('priceToBook')) else "N/A")
                    st.metric("Ex-Dividend Date", str(fundamental_data_yf.get('exDividendDate', "N/A")))

                st.subheader("Datas Importantes e Dividendos Recentes")
                earnings_dates_list = fundamental_data_yf.get('earningsDates', [])
                if earnings_dates_list and earnings_dates_list != ["N/A"]:
                    st.write(f"**Próximas Datas de Resultados (Estimadas):** {', '.join(map(str, earnings_dates_list))}")
                else:
                    st.write("**Próximas Datas de Resultados (Estimadas):** N/A")

                if fundamental_data_yf and fundamental_data_yf.get('recent_dividends'):
                    st.write("**Histórico Recente de Dividendos (Pagos):**")
                    df_divs = pd.DataFrame(fundamental_data_yf['recent_dividends'])
                    df_divs['Data'] = df_divs['Data'].astype(str) # Garantir que data seja string
                    df_divs['Valor'] = df_divs['Valor'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    st.dataframe(df_divs[['Data', 'Valor']], use_container_width=True, hide_index=True)
                else: st.write("**Histórico Recente de Dividendos:** N/A")

                with st.expander("Ver Mais Dados Fundamentalistas (Yahoo Finance - Raw)"):
                    if fundamental_data_yf: st.json(fundamental_data_yf, expanded=False)
                    else: st.write("Dados do Yahoo Finance não disponíveis.")
                if company_profile_fh:
                    with st.expander("Ver Perfil da Empresa (Finnhub - Raw)"):
                        st.json(company_profile_fh, expanded=False)
            else:
                st.warning("Dados fundamentalistas e de perfil não puderam ser carregados.")


        # --- Processar e Mostrar Resultados da Análise de Volatilidade ---
        if isinstance(analysis_result, str) and analysis_result.startswith("Erro:"): 
            with tab1: # Mostrar erro na aba principal
                st.error(analysis_result)
        elif isinstance(analysis_result, dict):
            st.success(f"Análise para {analysis_result['ticker']} concluída!")
            
            with tab1:
                st.header(f"Resultados da Análise para {analysis_result['ticker']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Métricas Principais")
                    st.metric("Preço Subjacente", f"{analysis_result.get('underlying_price', 0):.2f}" if pd.notna(analysis_result.get('underlying_price')) else "N/A")
                    st.metric("Vol. Histórica (YZ)", f"{analysis_result.get('historical_volatility_yz', 0):.2%}" if pd.notna(analysis_result.get('historical_volatility_yz')) else "N/A")
                    st.metric("Vol. Implícita ATM (30 DTE)", f"{analysis_result.get('iv30', 0):.2%}" if pd.notna(analysis_result.get('iv30')) else "N/A")
                    st.metric("Relação IV30/RV_YZ", f"{analysis_result.get('iv_rv_ratio', 0):.2f}" if pd.notna(analysis_result.get('iv_rv_ratio')) else "N/A")
                    st.metric("Volume Médio (30d)", f"{analysis_result.get('avg_volume', 0):,.0f}" if pd.notna(analysis_result.get('avg_volume')) else "N/A")
                    st.metric("Inclinação Estr. Termo (~0-45d)", f"{analysis_result.get('term_structure_slope_0_45', 0):.4f}" if pd.notna(analysis_result.get('term_structure_slope_0_45')) else "N/A")
                with col2:
                    st.subheader("Recomendação de Estratégia")
                    st.info(analysis_result.get('recommendation', "N/A"))
                    st.subheader("Exemplo Straddle ATM (Mais Próximo)")
                    s_info = analysis_result.get('straddle_example')
                    if s_info:
                        st.write(f"Expiração: {s_info.get('expiry', 'N/A')}, DTE: {s_info.get('days_to_expiry', 'N/A')}")
                        st.write(f"Call: Strike {s_info.get('call_strike', 0):.2f}, Preço {s_info.get('call_price', 0):.2f}")
                        st.write(f"Put: Strike {s_info.get('put_strike', 0):.2f}, Preço {s_info.get('put_price', 0):.2f}")
                        st.write(f"Custo Total: {s_info.get('total_cost', 0):.2f}, Mov. Esperado: ±{s_info.get('expected_move_pct', 0):.2f}%")
                    else: st.write("Não foi possível calcular exemplo de straddle ou dados insuficientes.")
                
                st.divider()
                st.subheader("Estrutura a Termo da Volatilidade Implícita (ATM)")
                fig_vol = plot_volatility_surface(
                    analysis_result.get('dtes_for_plot', []), 
                    analysis_result.get('ivs_for_plot', []), 
                    analysis_result.get('ticker', 'N/A'), 
                    analysis_result.get('atm_iv_data')
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            with tab2:
                st.header(f"Dados Históricos para {analysis_result.get('ticker', 'N/A')}")
                price_df = analysis_result.get('price_history_df')
                if price_df is not None and not price_df.empty:
                    st.dataframe(price_df.sort_index(ascending=False).head(20)) # Mostra os 20 mais recentes
                    csv_data = convert_df_to_csv(price_df)
                    st.download_button(label="Download Histórico Completo (CSV)", data=csv_data, 
                                       file_name=f"{analysis_result.get('ticker', 'ativo').replace('.', '_')}_historico_completo.csv", mime='text/csv')
                    st.subheader("Gráfico de Histórico de Preços")
                    fig_price = plot_price_history(price_df, analysis_result.get('ticker', 'N/A'))
                    st.plotly_chart(fig_price, use_container_width=True)
                else: st.warning("Dados de histórico de preços não disponíveis para exibição ou download.")
        else: 
            with tab1:
                st.error("Ocorreu um erro desconhecido durante a análise. Resultado não reconhecido.")
                logger.error(f"Resultado da análise não reconhecido para {ticker_input_processed}: {type(analysis_result)}")
        
        if hasattr(current_status_placeholder, 'empty'): current_status_placeholder.empty() 
        if progress_bar_container: progress_bar_container.empty() # Limpa o container e a barra


    with tab_about: 
        st.header("Sobre esta Aplicação")
        st.info("Este Analisador de Estratégias de Volatilidade foi desenvolvido para fins educacionais e de pesquisa. "
                "Utiliza dados de mercado de fontes como Alpha Vantage, yFinance e Finnhub.")
        st.subheader("Aviso Legal Importante")
        st.warning(DISCLAIMER_TEXT) # Definido no início do arquivo para fácil acesso
        
        st.subheader("Log da Aplicação")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'rb') as f: # Abrir em modo binário para download
                st.download_button(
                    label="Download do Arquivo de Log (analysis_log.txt)",
                    data=f,
                    file_name=LOG_FILE,
                    mime="text/plain"
                )
            with st.expander("Ver Últimas Entradas do Log (Máx. 50 linhas)"):
                try:
                    with open(LOG_FILE, 'r', encoding='utf-8') as f_read:
                        log_lines = f_read.readlines()
                        st.code("".join(log_lines[-50:]), language='log') # Mostra as últimas 50 linhas
                except Exception as e_log_read:
                    st.error(f"Não foi possível ler o arquivo de log: {e_log_read}")
        else:
            st.write("Arquivo de log ainda não foi criado.")

        st.markdown("--- ")
        st.markdown("Desenvolvido com auxílio de IA.")
        st.markdown("Desenvolvido com finalidade educacional apenas.")

DISCLAIMER_TEXT = """
Este software é fornecido apenas para fins educacionais e de pesquisa.
Não tem como objetivo fornecer aconselhamento de investimento, e nenhuma recomendação de investimento é feita aqui.
Os desenvolvedores não são consultores financeiros e não aceitam responsabilidade por quaisquer decisões financeiras
ou perdas resultantes do uso deste software. Sempre consulte um consultor financeiro profissional antes de
tomar qualquer decisão de investimento.
"""

if __name__ == "__main__":
    # Adicionar o disclaimer no início do log também
    logger.info(f"DISCLAIMER: {DISCLAIMER_TEXT.replace('\n', ' ')}")
    main()
