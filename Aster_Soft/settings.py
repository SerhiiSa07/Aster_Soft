EXCHANGE_PROFILE = "hibachi"  # Available profiles: "hibachi", "aster"
EXCHANGE = {
    # HIBACHI-CHANGE: configure one or more Hibachi REST endpoints.
    # If several URLs are provided the bot will try them in sequence on network errors.
    "base_urls": [
        "https://api.hibachi.xyz",
        "https://hibachi.xyz",
        "https://www.hibachi.xyz",
        # {"url": "https://203.0.113.10", "host": "api.hibachi.xyz", "verify_ssl": False},
        # ↑ HIBACHI-CHANGE: добавьте реальные IP-адреса Hibachi, если DNS недоступен.
    ],
    # The first URL in ``base_urls`` is also exposed as ``base_url`` for backward compatibility.
    "base_url": "https://api.hibachi.xyz",
    "origin": "https://hibachi.xyz",
    "referer": "https://hibachi.xyz/",
    "api_key_header": "Authorization",
    # Для REST-запросов Hibachi accountId передается как query-параметр, поэтому
    # отдельный заголовок не нужен.
    "account_header": None,
    "extra_headers": {
        "X-EXCHANGE": "hibachi",
    },
    # HIBACHI-CHANGE: автоматически добавляем поддомены для нескольких публичных зон Hibachi.
    "fallback_domains": [
        "hibachi.xyz",
        "hibachi.finance",
        "hibachi.exchange",
    ],
    # HIBACHI-CHANGE: используем публичные DNS-серверы, если локальный резолвер не справляется.
    "dns_servers": [
        "1.1.1.1",
        "8.8.8.8",
    ],
}

SHUFFLE_WALLETS     = True
RETRY               = 3

# --- GENERAL SETTINGS ---
THREADS             = 1  # Оставить 1 для стабильности

TOKENS_TO_TRADE     = {
    "SOL"           : {
        "prices"    : [100, 260],
        "leverage"  : [3, 4],
        "open_price": [0.0, 0.0],
    },
    "ETH"           : {
        "prices"    : [3000, 4800],
        "leverage"  : [3, 4],
        "open_price": [0.0, 0.0],
    },
    "BTC"           : {
        "prices"    : [100000, 130000],
        "leverage"  : [3, 4],
        "open_price": [0.0, 0.0],
    },
}

FUTURE_ACTIONS      = {
    "Long"          : True,
    "Short"         : True,  # Или выбери одно направление
}

TRADES_COUNT        = [15, 30]  # Уменьшил для меньшего риска

# --- ORDER SETTINGS ---
TRADE_AMOUNTS       = {
    "amount"        : [25, 40],  # Более консервативно
    "percent"       : [0, 0],
}

FUTURES_LIMITS      = {
    "close_previous"        : True,
    "price_diff_amount"     : [0.0, 0.0],
    "price_diff_percent"    : [0.8, 1.2],  # Более реалистично
    "to_wait"               : 5,
}

STOP_LOSS_SETTING   = {
    "enable"                : True,
    "loss_diff_amount"      : [0.0, 0.0],
    "loss_diff_percent"     : [1.2, 2.0],  # Увеличил для волатильности
}

CANCEL_ORDERS       = {
    "orders"        : True,
    "positions"     : True,
}

PAIR_SETTINGS       = {
    "pair_amount"   : [2, 3],  # Меньше одновременных позиций
    "position_hold" : [180, 300],  # 3-5 минут
}

# --- SLEEP SETTINGS ---
SLEEP_BETWEEN_OPEN_ORDERS  = [15, 25]
SLEEP_BETWEEN_CLOSE_ORDERS = [15, 25]
SLEEP_AFTER_FUTURE  = [150, 240]  # 2.5-4 минуты
SLEEP_AFTER_ACC     = [240, 360]  # 4-6 минут

TG_BOT_TOKEN        = ''
TG_USER_ID          = []
