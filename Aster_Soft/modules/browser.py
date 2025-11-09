from dataclasses import dataclass  # HIBACHI-CHANGE: structure REST hosts with metadata
from urllib.parse import urlencode, urlparse, urlunparse
from aiohttp import ClientError, ClientSession, ClientConnectorError
from decimal import Decimal
from hashlib import sha256
from loguru import logger
from time import time
import hmac
import random
import asyncio

# Імпортуємо налаштування
from settings import (
    SHUFFLE_WALLETS, RETRY, THREADS, TOKENS_TO_TRADE, FUTURE_ACTIONS,
    TRADES_COUNT, TRADE_AMOUNTS, FUTURES_LIMITS, STOP_LOSS_SETTING,
    CANCEL_ORDERS, PAIR_SETTINGS, SLEEP_BETWEEN_OPEN_ORDERS,
    SLEEP_BETWEEN_CLOSE_ORDERS, SLEEP_AFTER_FUTURE, SLEEP_AFTER_ACC,
    EXCHANGE_PROFILE, EXCHANGE,
)
from modules.utils import parse_api_key, ParsedAPIKey  # HIBACHI-CHANGE: reuse Hibachi parser

DEFAULT_PROFILES = {
    "aster": {
        "base_urls": ["https://fapi.asterdex.com"],
        "base_url": "https://fapi.asterdex.com",
        "origin": "https://www.asterdex.com",
        "referer": "https://www.asterdex.com/",
        "api_key_header": "X-MBX-APIKEY",
    },
    "hibachi": {
        "base_urls": ["https://fapi.hibachi.finance"],
        "base_url": "https://fapi.hibachi.finance",
        "origin": "https://www.hibachi.finance",
        "referer": "https://www.hibachi.finance/",
        "api_key_header": "X-MBX-APIKEY",
        # Most Hibachi deployments expect the explicit account id header
        "account_header": "X-ACCOUNT-ID",
    },
}

PROFILE_KEY = (EXCHANGE_PROFILE or "hibachi").lower()
if PROFILE_KEY not in DEFAULT_PROFILES:
    logger.opt(colors=True).warning(
        f"[!] Unknown exchange profile '{PROFILE_KEY}', falling back to 'hibachi'"
    )
    PROFILE_KEY = "hibachi"

DEFAULT_EXCHANGE_CONFIG = DEFAULT_PROFILES[PROFILE_KEY]

EXCHANGE_CONFIG = {**DEFAULT_EXCHANGE_CONFIG, **(EXCHANGE or {})}


@dataclass(frozen=True)
class RestHost:
    """Holds REST connection metadata for one Hibachi host."""

    url: str
    host_header: str | None = None
    verify_ssl: bool | None = True


def _as_bool(value, default: bool = True) -> bool:
    """HIBACHI-CHANGE: interpret truthy/falsey flags from config safely."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return default


def _generate_url_variants(url: str) -> list[str]:
    """Return a list of candidate REST URLs derived from the provided host."""

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return []

    variants: list[str] = []

    def _add_variant(variant_url: str) -> None:
        cleaned = variant_url.rstrip('/')
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    _add_variant(url)

    hostname = parsed.netloc
    if hostname.startswith("fapi."):
        bare_domain = hostname[len("fapi."):]
        if bare_domain:
            _add_variant(urlunparse(parsed._replace(netloc=f"api.{bare_domain}")))
            _add_variant(urlunparse(parsed._replace(netloc=bare_domain)))
            _add_variant(urlunparse(parsed._replace(netloc=f"www.{bare_domain}")))

    return variants


def _normalize_base_hosts(config: dict) -> list[RestHost]:
    raw_entries: list = []
    if "base_urls" in config and isinstance(config["base_urls"], (list, tuple)):
        raw_entries.extend(config["base_urls"])
    if config.get("base_url"):
        raw_entries.append(config["base_url"])
    if not raw_entries:
        fallback = DEFAULT_EXCHANGE_CONFIG.get("base_url")
        if fallback:
            raw_entries.append(fallback)

    hosts: list[RestHost] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            cleaned = entry.strip()
            if not cleaned:
                continue
            for variant in _generate_url_variants(cleaned):
                hosts.append(RestHost(url=variant))
            continue

        if not isinstance(entry, dict):
            continue

        # HIBACHI-CHANGE: allow explicit host header and SSL toggle per entry.
        url_value = entry.get("url") or entry.get("base_url") or entry.get("address")
        if not isinstance(url_value, str):
            continue

        cleaned_url = url_value.strip()
        if not cleaned_url:
            continue

        host_header = entry.get("host") or entry.get("host_header")
        verify_ssl = entry.get("verify_ssl")
        if verify_ssl is None:
            verify_ssl = entry.get("ssl")

        for variant in _generate_url_variants(cleaned_url):
            hosts.append(
                RestHost(
                    url=variant,
                    host_header=host_header.strip() if isinstance(host_header, str) else None,
                    verify_ssl=_as_bool(verify_ssl, True),
                )
            )

    unique: list[RestHost] = []
    seen = set()
    for host in hosts:
        key = (host.url, host.host_header, host.verify_ssl)
        if key not in seen:
            seen.add(key)
            unique.append(host)

    if not unique:
        return [RestHost(url="https://fapi.hibachi.finance")]

    return unique


BASE_HOSTS = _normalize_base_hosts(EXCHANGE_CONFIG)
BASE_URL = BASE_HOSTS[0].url
ORIGIN = EXCHANGE_CONFIG.get("origin")
REFERER = EXCHANGE_CONFIG.get("referer")
API_KEY_HEADER_NAME = EXCHANGE_CONFIG.get("api_key_header", DEFAULT_EXCHANGE_CONFIG["api_key_header"])
ACCOUNT_HEADER_NAME = EXCHANGE_CONFIG.get("account_header")
EXTRA_HEADERS = EXCHANGE_CONFIG.get("extra_headers", {})

DEFAULT_ENDPOINTS = {
    "time": "/fapi/v1/time",
    "exchange_info": "/fapi/v1/exchangeInfo",
    "order": "/fapi/v1/order",
    "balance": "/fapi/v2/balance",
    "ticker_price": "/fapi/v1/ticker/price",
    "account": "/fapi/v4/account",
    "leverage": "/fapi/v1/leverage",
    "position_risk": "/fapi/v2/positionRisk",
    "open_orders": "/fapi/v1/openOrders",
    "cancel_all": "/fapi/v1/allOpenOrders",
    "order_book": "/fapi/v1/depth",
}

ENDPOINTS = {**DEFAULT_ENDPOINTS, **EXCHANGE_CONFIG.get("endpoints", {})}


class Browser:
    BASE_URL: str = BASE_URL
    BASE_HOSTS: list[RestHost] = BASE_HOSTS
    API_KEY_HEADER_NAME: str = API_KEY_HEADER_NAME
    ACCOUNT_HEADER_NAME: str | None = ACCOUNT_HEADER_NAME
    ORIGIN: str | None = ORIGIN
    REFERER: str | None = REFERER
    ENDPOINTS: dict[str, str] = ENDPOINTS

    def __init__(self, proxy: str, api_key: str, label: str, account_reference: str | None = None):
        self.max_retries = RETRY
        self.base_hosts = list(self.BASE_HOSTS)
        if not self.base_hosts:
            self.base_hosts = [RestHost(url=BASE_URL)]
        self._base_url_index = 0
        self.api_key = api_key
        self.label = label
        self.endpoints = dict(self.ENDPOINTS)

        (
            self.api_key_header_value,
            self.api_secret_raw,
            parsed_account_reference,
            self._parsed_credentials,
        ) = self._extract_credentials(api_key, account_reference)  # HIBACHI-CHANGE: align browser credentials with Hibachi parsing
        if account_reference and parsed_account_reference and account_reference != parsed_account_reference:
            logger.opt(colors=True).warning(
                f'[!] <white>{self.label}</white> | Account reference mismatch: '
                f'<white>{parsed_account_reference}</white> (parsed) vs '
                f'<white>{account_reference}</white> (provided)'
            )
        self.account_reference = account_reference or parsed_account_reference
        self.api_secret_bytes = self._normalize_secret(self.api_secret_raw)
        self.api_key_header_name = self.API_KEY_HEADER_NAME
        self.account_header_name = self.ACCOUNT_HEADER_NAME
        self.extra_headers = dict(EXTRA_HEADERS)

        if self.account_reference:
            logger.opt(colors=True).debug(
                f'[•] <white>{self.label}</white> | Account reference <white>{self.account_reference}</white>'
            )
        if getattr(self, "_parsed_credentials", None) and self._parsed_credentials.wallet_address:
            logger.opt(colors=True).debug(
                f"[•] <white>{self.label}</white> | HIBACHI-CHANGE wallet <white>{self._parsed_credentials.wallet_address}</white>"
            )

        if proxy not in ['https://log:pass@ip:port', 'http://log:pass@ip:port', 'log:pass@ip:port', '', None]:
            cleaned_proxy = proxy
            if not proxy.startswith(("http://", "https://")):
                cleaned_proxy = f"http://{proxy}"
            self.proxy = cleaned_proxy
            logger.opt(colors=True).debug(f'[•] <white>{self.label}</white> | Using proxy <white>{self.proxy}</white>')
        else:
            self.proxy = None
            logger.opt(colors=True).info(f'[•] <white>{self.label}</white> | Using direct connection (no proxy)')

        self._base_headers = self._build_base_headers()
        self.session = self.get_new_session()

    def _build_base_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        }
        if self.ORIGIN:
            headers["Origin"] = self.ORIGIN
        if self.REFERER:
            headers["Referer"] = self.REFERER
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def get_new_session(self):
        return ClientSession(headers=self._base_headers)

    async def close_session(self):
        if self.session:
            await self.session.close()

    def _extract_credentials(
        self,
        api_key: str,
        provided_account_reference: str | None = None,
    ) -> tuple[str, str, str | None, ParsedAPIKey]:
        if provided_account_reference:
            normalized_reference = provided_account_reference.strip()
            prefixed_key = api_key
            if normalized_reference and not api_key.startswith(f"{normalized_reference}:"):
                prefixed_key = f"{normalized_reference}:{api_key}"
        else:
            prefixed_key = api_key

        parsed = parse_api_key(prefixed_key, default_label=self.label)  # HIBACHI-CHANGE: single parsing routine
        account_reference = parsed.account_id
        if not account_reference and len(parsed.credential_parts) > 2:
            account_reference = parsed.credential_parts[-3]
        return parsed.api_key, parsed.secret_key, account_reference, parsed

    @staticmethod
    def _normalize_secret(secret: str) -> bytes:
        if secret.startswith("0x"):
            hex_part = secret[2:]
            if len(hex_part) % 2:
                hex_part = f"0{hex_part}"
            try:
                return bytes.fromhex(hex_part)
            except ValueError:
                masked = f"{hex_part[:4]}…" if len(hex_part) > 8 else "****"
                logger.opt(colors=True).warning(
                    f'[!] Failed to parse hex secret ({masked}), falling back to raw bytes'
                )
        return secret.encode('utf-8')

    def _current_host(self) -> RestHost:
        return self.base_hosts[self._base_url_index]

    def _advance_base_url(self) -> None:
        if len(self.base_hosts) <= 1:
            return
        previous = self._current_host()
        self._base_url_index = (self._base_url_index + 1) % len(self.base_hosts)
        current = self._current_host()
        if current.url != previous.url:
            logger.opt(colors=True).warning(
                f'[!] <white>{self.label}</white> | Switching REST host to <white>{current.url}</white>'
            )

    def _build_url(self, endpoint_key: str) -> str:
        path = self.endpoints.get(endpoint_key)
        if path is None:
            raise ValueError(f"Unknown endpoint '{endpoint_key}'")
        return f'{self._current_host().url}{path}'

    def _build_signature(self, all_params: dict, method: str):
        query_string = urlencode(all_params)

        signature = hmac.new(
            self.api_secret_bytes,
            query_string.encode('utf-8'),
            sha256
        ).hexdigest()

        headers = {self.api_key_header_name: self.api_key_header_value}
        if self.account_header_name and self.account_reference:
            headers[self.account_header_name] = self.account_reference
        if self.extra_headers:
            headers.update(self.extra_headers)

        return {
            "headers": headers,
            "data" if method == "POST" else "params": {
                "signature": signature
            }
        }

    async def send_request(self, endpoint: str | None = None, **kwargs):
        for attempt in range(self.max_retries):
            try:
                current_host = self._current_host()
                if endpoint:
                    kwargs["url"] = self._build_url(endpoint)
                if kwargs.get("method"):
                    kwargs["method"] = kwargs["method"].upper()
                if self.proxy:
                    kwargs["proxy"] = self.proxy
                if current_host.verify_ssl is not None:
                    kwargs.setdefault("ssl", current_host.verify_ssl)

                if kwargs.get("build_signature"):
                    del kwargs["build_signature"]

                    # Get server time first
                    try:
                        async with self.session.request(
                            method="GET",
                            url=self._build_url("time"),
                            proxy=self.proxy
                        ) as time_req:
                            time_data = await time_req.json()
                            server_time = time_data["serverTime"]
                    except:
                        server_time = int(time() * 1000)

                    if kwargs.get("params") is None:
                        kwargs["params"] = {}
                    kwargs["params"].update({
                        "timestamp": server_time,
                        "recvWindow": 10000,
                    })

                    if kwargs["method"] == "POST":
                        all_params = {**kwargs.get("params", {}), **kwargs.get("data", {})}
                        kwargs["data"] = all_params
                        if kwargs.get("params"): 
                            del kwargs["params"]
                    else:
                        all_params = kwargs.get("params", {})

                    for k, v in self._build_signature(all_params, kwargs["method"]).items():
                        if kwargs.get(k) is None:
                            kwargs[k] = v
                        else:
                            kwargs[k].update(v)

                if current_host.host_header:
                    kwargs.setdefault("headers", {})
                    kwargs["headers"].setdefault("Host", current_host.host_header)

                async with self.session.request(**kwargs) as response:
                    data = await response.json()
                    
                    if 'code' in data and data['code'] != 200:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise Exception(f"API Error: {data}")
                    
                    return data
                    
            except Exception as e:
                if isinstance(e, (ClientConnectorError, ClientError, asyncio.TimeoutError)):
                    self._advance_base_url()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise e

    async def get_tokens_data(self):
        response = await self.send_request(
            method="GET",
            endpoint="exchange_info",
        )
        if response.get("symbols") is None:
            raise Exception(f'Failed to get tokens data: {response}')
        return response["symbols"]

    async def create_order(self, order_data: dict):
        response = await self.send_request(
            method="POST",
            endpoint="order",
            data=order_data,
            build_signature=True,
        )
        if response.get("orderId") is None:
            raise Exception(f'Failed to create order: {response}')
        return response

    async def get_balance(self):
        response = await self.send_request(
            method="GET",
            endpoint="balance",
            build_signature=True,
        )
        if type(response) is not list:
            raise Exception(f'Failed to get balance: {response}')

        return next((float(token["availableBalance"]) for token in response if token["asset"] == "USDT"), 0)

    async def get_token_price(self, token_name: str):
        response = await self.send_request(
            method="GET",
            endpoint="ticker_price",
            params={"symbol": token_name + "USDT"},
        )
        if response.get("price") is None:
            raise Exception(f'Failed to get {token_name} price: {response}')

        return Decimal(response["price"])

    async def get_leverages(self):
        response = await self.send_request(
            method="GET",
            endpoint="account",
            build_signature=True,
        )
        if response.get("positions") is None:
            raise Exception(f'Failed to get leverages: {response}')

        return {p["symbol"].removesuffix("USDT"): int(p["leverage"]) for p in response["positions"]}

    async def change_leverage(self, token_name: str, leverage: int):
        response = await self.send_request(
            method="POST",
            endpoint="leverage",
            data={
                "symbol": f"{token_name}USDT",
                "leverage": leverage,
            },
            build_signature=True,
        )
        if response.get("leverage") != leverage:
            raise Exception(f'Failed to change leverage: {response}')

        return True

    async def get_account_positions(self):
        response = await self.send_request(
            method="GET",
            endpoint="position_risk",
            build_signature=True,
        )
        if type(response) is not list:
            raise Exception(f'Failed to get account positions: {response}')

        return [p for p in response if Decimal(p["positionAmt"])]

    async def get_account_orders(self):
        response = await self.send_request(
            method="GET",
            endpoint="open_orders",
            build_signature=True,
        )
        if type(response) is not list:
            raise Exception(f'Failed to get account orders: {response}')

        return response

    async def close_all_open_orders(self, token_name: str):
        response = await self.send_request(
            method="DELETE",
            endpoint="cancel_all",
            params={"symbol": f"{token_name}USDT"},
            build_signature=True,
        )
        if response != {'code': 200, 'msg': 'The operation of cancel all open order is done.'}:
            raise Exception(f'Failed to close {token_name} orders: {response}')

        return True

    async def get_token_order_book(self, token_name: str):
        response = await self.send_request(
            method="GET",
            endpoint="order_book",
            params={"symbol": f"{token_name}USDT", "limit": 50},
        )
        if response.get("bids") is None or response.get("asks") is None:
            raise Exception(f'Failed to get {token_name} order book: {response}')

        return {"BUY": float(response["bids"][0][0]), "SELL": float(response["asks"][0][0])}


class TradingBot:
    def __init__(self, browser: Browser, wallet_label: str):
        self.browser = browser
        self.wallet_label = wallet_label
        self.trades_count = 0
        self.max_trades = random.randint(*TRADES_COUNT)

    async def random_sleep(self, sleep_range: list):
        delay = random.uniform(*sleep_range)
        logger.info(f"[{self.wallet_label}] Sleeping for {delay:.2f}s")
        await asyncio.sleep(delay)

    async def execute_trade(self, symbol: str, action: str):
        try:
            balance = await self.browser.get_balance()
            if balance < 10:
                logger.error(f"[{self.wallet_label}] Insufficient balance: ${balance:.2f}")
                return False

            # Calculate trade amount
            if TRADE_AMOUNTS["amount"][1] > 0:
                trade_amount = random.uniform(*TRADE_AMOUNTS["amount"])
            else:
                percent = random.uniform(*TRADE_AMOUNTS["percent"])
                trade_amount = balance * percent / 100

            # Set leverage
            leverage = random.randint(*TOKENS_TO_TRADE[symbol]["leverage"])
            await self.browser.change_leverage(symbol, leverage)

            # Get current price
            orderbook = await self.browser.get_token_order_book(symbol)
            current_price = orderbook["BUY"] if action == "Long" else orderbook["SELL"]

            # Calculate position size
            position_size = trade_amount * leverage / current_price

            # Create order
            order_data = {
                "symbol": f"{symbol}USDT",
                "side": "BUY" if action == "Long" else "SELL",
                "type": "MARKET",
                "quantity": round(position_size, 4),
            }

            await self.browser.create_order(order_data)

            logger.success(
                f"[{self.wallet_label}] {action} {position_size:.4f} {symbol} "
                f"at ${current_price:.2f} (${trade_amount:.2f}, {leverage}x)"
            )

            return True

        except Exception as e:
            logger.error(f"[{self.wallet_label}] Trade execution failed: {e}")
            return False

    async def close_all_positions(self, symbol: str = None):
        try:
            positions = await self.browser.get_account_positions()
            for position in positions:
                pos_symbol = position["symbol"].replace("USDT", "")
                if symbol and pos_symbol != symbol:
                    continue

                position_amt = Decimal(position["positionAmt"])
                if position_amt != 0:
                    side = "SELL" if position_amt > 0 else "BUY"
                    order_data = {
                        "symbol": position["symbol"],
                        "side": side,
                        "type": "MARKET",
                        "quantity": abs(float(position_amt))
                    }
                    await self.browser.create_order(order_data)
                    logger.info(f"[{self.wallet_label}] Closed {pos_symbol} position")

            return True
        except Exception as e:
            logger.error(f"[{self.wallet_label}] Error closing positions: {e}")
            return False

    async def run_trading_cycle(self):
        logger.info(f"[{self.wallet_label}] Starting trading cycle")

        try:
            available_symbols = list(TOKENS_TO_TRADE.keys())
            
            while self.trades_count < self.max_trades:
                symbol = random.choice(available_symbols)
                available_actions = [action for action, enabled in FUTURE_ACTIONS.items() if enabled]
                
                if not available_actions:
                    logger.error(f"[{self.wallet_label}] No trading actions enabled!")
                    break
                
                action = random.choice(available_actions)
                
                success = await self.execute_trade(symbol, action)
                
                if success:
                    self.trades_count += 1
                    
                    hold_time = random.randint(*PAIR_SETTINGS["position_hold"])
                    logger.info(f"[{self.wallet_label}] Holding position for {hold_time}s")
                    await asyncio.sleep(hold_time)
                    
                    await self.close_all_positions(symbol)
                    
                    if self.trades_count < self.max_trades:
                        await self.random_sleep(SLEEP_AFTER_FUTURE)
                else:
                    await self.random_sleep(SLEEP_BETWEEN_OPEN_ORDERS)
            
            logger.success(f"[{self.wallet_label}] Trading cycle completed: {self.trades_count}/{self.max_trades} trades")

        except Exception as e:
            logger.error(f"[{self.wallet_label}] Trading cycle failed: {e}")


async def process_wallet(wallet_data: dict, wallet_id: int):
    label = f"Wallet_{wallet_id}"
    browser = None
    
    try:
        browser = Browser(
            proxy=wallet_data.get("proxy", ""),
            api_key=wallet_data["api_key"],
            label=label
        )
        
        balance = await browser.get_balance()
        logger.info(f"[{label}] Connected successfully. Balance: ${balance:.2f}")
        
        bot = TradingBot(browser, label)
        await bot.run_trading_cycle()
        
        await asyncio.sleep(random.uniform(*SLEEP_AFTER_ACC))
        
    except Exception as e:
        logger.error(f"[{label}] Critical error: {e}")
    finally:
        if browser:
            await browser.close_session()


async def main():
    WALLETS = [
        {
            "api_key": "your_api_key_here:your_secret_key_here",
            "proxy": "username:password@ip:port"  # optional
        },
        # Add more wallets here...
    ]
    
    if not WALLETS:
        logger.error("No wallets configured!")
        return
    
    if SHUFFLE_WALLETS:
        random.shuffle(WALLETS)
        logger.info("Wallets shuffled")
    
    logger.info(f"Starting trading bot with {len(WALLETS)} wallets")
    
    tasks = []
    for i, wallet_data in enumerate(WALLETS):
        if THREADS > 1:
            task = asyncio.create_task(process_wallet(wallet_data, i + 1))
            tasks.append(task)
            
            if len(tasks) >= THREADS:
                await asyncio.gather(*tasks)
                tasks = []
        else:
            await process_wallet(wallet_data, i + 1)
    
    if tasks:
        await asyncio.gather(*tasks)
    
    logger.success("All wallets processed!")


if __name__ == "__main__":
    logger.add("trading_bot.log", rotation="10 MB", level="INFO")
    asyncio.run(main())