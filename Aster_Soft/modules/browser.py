from dataclasses import dataclass  # HIBACHI-CHANGE: structure REST hosts with metadata
from urllib.parse import urlparse, urlunparse
from aiohttp import ClientError, ClientSession, ClientConnectorError, TCPConnector
try:  # HIBACHI-CHANGE: support custom DNS resolvers when aiohttp exposes them.
    from aiohttp.resolver import AsyncResolver
except Exception:  # pragma: no cover - AsyncResolver is optional in some aiohttp builds.
    AsyncResolver = None
_resolver_warning_logged = False  # HIBACHI-CHANGE: avoid spamming warnings when aiodns is missing.
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from loguru import logger
from time import time
import hashlib
import random
import hmac
import asyncio

from eth_account import Account

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
        "fallback_prefixes": ["fapi"],
    },
    "hibachi": {
        "base_urls": [
            "https://api.hibachi.xyz",
            "https://hibachi.xyz",
            "https://www.hibachi.xyz",
        ],
        "base_url": "https://api.hibachi.xyz",
        "origin": "https://hibachi.xyz",
        "referer": "https://hibachi.xyz/",
        "api_key_header": "Authorization",
        # Hibachi передает accountId как query-параметр, поэтому отдельный заголовок не обязателен.
        "account_header": None,
        # HIBACHI-CHANGE: common public domains to try if DNS is missing records.
        "fallback_domains": [
            "hibachi.xyz",
            "hibachi.finance",
            "hibachi.exchange",
        ],
        "fallback_prefixes": ["api", None],
        # HIBACHI-CHANGE: use public DNS servers by default to avoid local resolver issues.
        "dns_servers": ["1.1.1.1", "8.8.8.8"],
        # HIBACHI-CHANGE: separate host for public market data endpoints.
        "data_base_url": "https://data-api.hibachi.xyz",
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
    netloc = parsed.netloc or ""
    if not parsed.scheme or not netloc:
        return []

    cleaned_url = url.rstrip('/')
    variants: list[str] = [cleaned_url]

    hostname = parsed.hostname or ""
    if hostname and not hostname.startswith("www."):
        www_variant = urlunparse(parsed._replace(netloc=f"www.{hostname}")).rstrip('/')
        if www_variant not in variants:
            variants.append(www_variant)

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

    fallback_domains = config.get("fallback_domains") or []
    if isinstance(fallback_domains, (list, tuple)):
        scheme_hint = "https"
        if hosts:
            parsed = urlparse(hosts[0].url)
            if parsed.scheme:
                scheme_hint = parsed.scheme
        fallback_prefixes = config.get("fallback_prefixes")
        if isinstance(fallback_prefixes, str):
            fallback_prefixes = [fallback_prefixes]
        elif not isinstance(fallback_prefixes, (list, tuple)):
            fallback_prefixes = [None]

        for domain in fallback_domains:
            if not isinstance(domain, str):
                continue
            cleaned_domain = domain.strip().lstrip(".")
            if not cleaned_domain:
                continue
            for prefix in fallback_prefixes:
                prefix = prefix.strip() if isinstance(prefix, str) else None
                if prefix:
                    primary_url = f"{scheme_hint}://{prefix}.{cleaned_domain}"
                else:
                    primary_url = f"{scheme_hint}://{cleaned_domain}"
                for variant in _generate_url_variants(primary_url):
                    hosts.append(RestHost(url=variant))

    unique: list[RestHost] = []
    seen = set()
    for host in hosts:
        key = (host.url, host.host_header, host.verify_ssl)
        if key not in seen:
            seen.add(key)
            unique.append(host)

    if not unique:
        return [RestHost(url="https://api.hibachi.xyz")]

    return unique


BASE_HOSTS = _normalize_base_hosts(EXCHANGE_CONFIG)
BASE_URL = BASE_HOSTS[0].url
DATA_BASE_URL = EXCHANGE_CONFIG.get("data_base_url") or DEFAULT_EXCHANGE_CONFIG.get("data_base_url") or "https://data-api.hibachi.xyz"
ORIGIN = EXCHANGE_CONFIG.get("origin")
REFERER = EXCHANGE_CONFIG.get("referer")
API_KEY_HEADER_NAME = EXCHANGE_CONFIG.get("api_key_header", DEFAULT_EXCHANGE_CONFIG["api_key_header"])
ACCOUNT_HEADER_NAME = EXCHANGE_CONFIG.get("account_header")
EXTRA_HEADERS = EXCHANGE_CONFIG.get("extra_headers", {})
DNS_NAMESERVERS = EXCHANGE_CONFIG.get("dns_servers")

DEFAULT_ENDPOINTS = {
    "time": {"path": "/exchange/utc-timestamp", "base": "data"},
    "exchange_info": {"path": "/market/exchange-info", "base": "data"},
    "order": {"path": "/trade/order", "base": "trade"},
    "balance": {"path": "/capital/balance", "base": "trade"},
    "ticker_price": {"path": "/market/data/prices", "base": "data"},
    "account": {"path": "/trade/account/info", "base": "trade"},
    "leverage": {"path": "/trade/account/leverage", "base": "trade"},
    "position_risk": {"path": "/trade/account/info", "base": "trade"},
    "open_orders": {
        "paths": [
            "/trade/account/orders",
            "/trade/orders",
        ],
        "base": "trade",
    },
    "cancel_all": {"path": "/trade/orders", "base": "trade"},
    "order_book": {"path": "/market/data/orderbook", "base": "data"},
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
        if len(self.base_hosts) > 1:
            host_urls = ", ".join(host.url for host in self.base_hosts)
            logger.opt(colors=True).debug(
                f"[•] <white>{label}</white> | REST hosts rotation: <white>{host_urls}</white>"
            )
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
        self.account_id_value = None
        if self.account_reference:
            ref_segments = [seg for seg in self.account_reference.split(":") if seg]
            if ref_segments:
                self.account_id_value = ref_segments[-1]

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
        self._contracts_by_symbol: dict[str, dict] = {}
        self._fee_config: dict = {}
        self._tokens_cache: list[dict] | None = None

    def _build_base_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        if self.ORIGIN:
            headers["Origin"] = self.ORIGIN
        if self.REFERER:
            headers["Referer"] = self.REFERER
        if self.API_KEY_HEADER_NAME and self.api_key_header_value:
            headers[self.API_KEY_HEADER_NAME] = self.api_key_header_value
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def get_new_session(self):
        # HIBACHI-CHANGE: trust environment variables so system-wide proxy or DNS settings apply.
        connector = self._build_connector()
        if connector is not None:
            return ClientSession(headers=self._base_headers, trust_env=True, connector=connector)
        return ClientSession(headers=self._base_headers, trust_env=True)

    @staticmethod
    def _normalize_symbol(symbol: str | None) -> str:
        if not symbol:
            return ""
        cleaned = symbol.strip().upper()
        if not cleaned:
            return ""
        if cleaned.endswith("USDT-P"):
            if "/" not in cleaned:
                base = cleaned[:-7]
                return f"{base}/USDT-P"
            return cleaned
        if cleaned.endswith("USDT"):
            base = cleaned[:-4]
            return f"{base}/USDT-P"
        if "/" not in cleaned:
            return f"{cleaned}/USDT-P"
        return cleaned

    @staticmethod
    def _token_from_symbol(symbol: str | None) -> str:
        normalized = Browser._normalize_symbol(symbol)
        if not normalized:
            return ""
        if normalized.endswith("/USDT-P"):
            return normalized[:-8]
        if normalized.endswith("/USDT"):
            return normalized[:-6]
        return normalized

    @staticmethod
    def _decimal_precision_from_step(step: str | None) -> int:
        if not step:
            return 0
        step = step.strip()
        if not step or step == "1":
            return 0
        if "." not in step:
            return 0
        decimals = step.split(".")[1]
        return len(decimals.rstrip("0"))

    @staticmethod
    def _quantize_decimal(value: Decimal, precision: int) -> Decimal:
        if precision <= 0:
            return value.to_integral_value(rounding=ROUND_DOWN)
        exp = Decimal(1).scaleb(-precision)
        return value.quantize(exp, rounding=ROUND_DOWN)

    @staticmethod
    def _format_decimal(value: Decimal, precision: int) -> str:
        quantized = Browser._quantize_decimal(value, precision)
        if precision <= 0:
            return str(int(quantized))
        return f"{quantized:.{precision}f}".rstrip("0").rstrip(".")

    @staticmethod
    def _pack_uint(value: int, length: int) -> bytes:
        if value < 0:
            raise ValueError("Value for signing must be non-negative")
        return int(value).to_bytes(length, "big", signed=False)

    def _sign_payload(self, payload: bytes) -> str:
        if not payload:
            raise ValueError("Payload for signing cannot be empty")
        if isinstance(self.api_secret_raw, str) and self.api_secret_raw.startswith("0x"):
            digest = hashlib.sha256(payload).digest()
            signer = Account.from_key(self.api_secret_raw)
            signed = signer.signHash(digest)
            signature_bytes = (
                int(signed.r).to_bytes(32, "big")
                + int(signed.s).to_bytes(32, "big")
                + bytes([signed.v - 27])
            )
            return signature_bytes.hex()
        return hmac.new(self.api_secret_bytes, payload, hashlib.sha256).hexdigest()

    def _account_id(self) -> str | None:
        return self.account_id_value or self.account_reference

    async def _ensure_tokens_cached(self) -> None:
        if self._tokens_cache is not None and self._contracts_by_symbol:
            return
        await self.get_tokens_data()

    def _lookup_contract(self, symbol: str | None) -> dict | None:
        normalized = self._normalize_symbol(symbol)
        if not normalized:
            return None
        return self._contracts_by_symbol.get(normalized)

    def _open_orders_candidates(self) -> list[tuple[str, str]]:
        """Return (base, path) tuples to query open orders, respecting overrides."""

        spec = self.endpoints.get("open_orders")
        base = "trade"
        raw_candidates: list = []

        def _append_path(value):
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    raw_candidates.append(cleaned)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _append_path(item)

        if isinstance(spec, dict):
            base = spec.get("base", "trade") or "trade"
            _append_path(spec.get("paths"))
            _append_path(spec.get("path"))
            _append_path(spec.get("url"))
        elif isinstance(spec, (list, tuple, set)):
            _append_path(spec)
        elif isinstance(spec, str):
            _append_path(spec)

        if not raw_candidates:
            raw_candidates = [
                "/trade/account/orders",
                "/trade/orders",
            ]

        seen: set[tuple[str, str]] = set()
        result: list[tuple[str, str]] = []
        for candidate in raw_candidates:
            key = (base, candidate)
            if key in seen:
                continue
            seen.add(key)
            result.append((base, candidate))

        return result

    def _resolve_base_url(self, base: str, path: str) -> str:
        cleaned = path.strip()
        if not cleaned:
            raise ValueError("Open-orders endpoint path is empty")

        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned

        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"

        if base == "data":
            return f"{DATA_BASE_URL}{cleaned}"

        return f"{self._current_host().url}{cleaned}"

    def _generate_nonce(self) -> int:
        return int(time() * 1_000_000)

    def _build_connector(self) -> TCPConnector | None:
        """HIBACHI-CHANGE: configure custom DNS servers for aiohttp if provided."""

        dns_servers = DNS_NAMESERVERS
        if AsyncResolver is None:
            global _resolver_warning_logged
            if dns_servers and not _resolver_warning_logged:
                logger.opt(colors=True).info(
                    f"[•] <white>{self.label}</white> | aiohttp.AsyncResolver unavailable; install 'aiodns' to enable custom DNS"
                )
                _resolver_warning_logged = True
            return None

        if not isinstance(dns_servers, (list, tuple)):
            return None

        servers: list[str] = []
        for server in dns_servers:
            if server is None:
                continue
            cleaned = str(server).strip()
            if cleaned:
                servers.append(cleaned)

        if not servers:
            return None

        try:
            resolver = AsyncResolver(nameservers=servers)
        except Exception as exc:  # pragma: no cover - depends on system resolver
            logger.opt(colors=True).warning(
                f"[!] <white>{self.label}</white> | Failed to initialise custom DNS resolver: {exc}"
            )
            return None

        logger.opt(colors=True).debug(
            f"[•] <white>{self.label}</white> | Using DNS servers <white>{', '.join(servers)}</white>"
        )
        return TCPConnector(resolver=resolver, ttl_dns_cache=300)

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
        spec = self.endpoints.get(endpoint_key)
        if spec is None:
            raise ValueError(f"Unknown endpoint '{endpoint_key}'")

        base = "trade"
        path = spec
        if isinstance(spec, dict):
            base = spec.get("base", "trade")
            path = spec.get("path") or spec.get("url")

        if not isinstance(path, str):
            raise ValueError(f"Endpoint '{endpoint_key}' is misconfigured: {spec}")

        cleaned_path = path.strip()
        if not cleaned_path:
            raise ValueError(f"Endpoint '{endpoint_key}' has empty path")

        if cleaned_path.startswith("http://") or cleaned_path.startswith("https://"):
            return cleaned_path

        if not cleaned_path.startswith("/"):
            cleaned_path = f"/{cleaned_path}"

        if base == "data":
            return f"{DATA_BASE_URL}{cleaned_path}"

        return f"{self._current_host().url}{cleaned_path}"

    async def send_request(self, endpoint: str | None = None, **kwargs):
        for attempt in range(self.max_retries):
            try:
                current_host = self._current_host()
                if endpoint:
                    kwargs["url"] = self._build_url(endpoint)

                method = kwargs.get("method", "GET").upper()
                kwargs["method"] = method

                if self.proxy:
                    kwargs["proxy"] = self.proxy
                if current_host.verify_ssl is not None:
                    kwargs.setdefault("ssl", current_host.verify_ssl)

                kwargs.pop("build_signature", None)

                if method == "GET":
                    params = kwargs.setdefault("params", {})
                else:
                    params = kwargs.get("params")

                if endpoint in {"balance", "account", "position_risk", "open_orders", "order"} and self.account_id_value:
                    params = kwargs.setdefault("params", {})
                    params.setdefault("accountId", self.account_id_value)

                if endpoint in {"cancel_all", "leverage"} and self.account_id_value:
                    body = kwargs.get("data") or {}
                    if not isinstance(body, dict):
                        body = {}
                    body.setdefault("accountId", self.account_id_value)
                    kwargs["data"] = body

                if method in {"POST", "PUT", "DELETE"}:
                    payload = kwargs.pop("data", None)
                    if payload is not None:
                        if not isinstance(payload, dict):
                            raise ValueError(f"Request body must be a dict, got {type(payload)}")
                        kwargs.setdefault("headers", {})
                        kwargs["headers"].setdefault("Content-Type", "application/json")
                        kwargs["json"] = payload
                    kwargs.pop("params", None)

                if current_host.host_header:
                    kwargs.setdefault("headers", {})
                    kwargs["headers"].setdefault("Host", current_host.host_header)

                async with self.session.request(**kwargs) as response:
                    raw_text = await response.text()

                    if response.status >= 400:
                        raise Exception(
                            f"HTTP {response.status} calling {kwargs.get('url')}: {raw_text or response.reason}"
                        )

                    if not raw_text:
                        return {}

                    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    if content_type and "json" not in content_type:
                        raise Exception(
                            f"Attempt to decode JSON with unexpected mimetype: {content_type}"
                        )

                    try:
                        return await response.json()
                    except Exception as exc:
                        raise Exception(f"Failed to parse JSON response: {raw_text}") from exc

            except Exception as e:
                if isinstance(e, (ClientConnectorError, ClientError, asyncio.TimeoutError)):
                    self._advance_base_url()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise e

    async def get_tokens_data(self):
        if self._tokens_cache is not None:
            return self._tokens_cache
        response = await self.send_request(
            method="GET",
            endpoint="exchange_info",
        )
        contracts = response.get("futureContracts")
        if not isinstance(contracts, list):
            raise Exception(f'Failed to get tokens data: {response}')
        self._fee_config = response.get("feeConfig") or {}
        self._contracts_by_symbol.clear()
        for contract in contracts:
            symbol = contract.get("symbol")
            normalized = self._normalize_symbol(symbol)
            if not normalized:
                continue
            self._contracts_by_symbol[normalized] = contract
            base_token = self._token_from_symbol(symbol)
            if base_token:
                self._contracts_by_symbol[base_token] = contract
                self._contracts_by_symbol[f"{base_token}USDT"] = contract
                self._contracts_by_symbol[f"{base_token}/USDT-P"] = contract
        self._tokens_cache = contracts
        return contracts

    async def create_order(self, order_data: dict):
        await self._ensure_tokens_cached()

        if not isinstance(order_data, dict):
            raise ValueError("Order data must be a dictionary")

        metadata = order_data.get("token_metadata") if isinstance(order_data.get("token_metadata"), dict) else None
        symbol_input = order_data.get("symbol") or (metadata.get("symbol") if metadata else None)
        normalized_symbol = self._normalize_symbol(symbol_input)
        if not normalized_symbol:
            raise Exception(f"Unable to determine Hibachi symbol for order: {order_data}")

        contract_info = metadata or self._lookup_contract(normalized_symbol)
        if not contract_info:
            raise Exception(f"Contract metadata not found for symbol {normalized_symbol}")

        contract_id = int(contract_info.get("contract_id") or contract_info.get("id"))
        underlying_decimals = int(contract_info.get("underlying_decimals") or contract_info.get("underlyingDecimals") or 0)
        settlement_decimals = int(contract_info.get("settlement_decimals") or contract_info.get("settlementDecimals") or 0)
        size_precision = int(contract_info.get("size") or contract_info.get("size_precision") or self._decimal_precision_from_step(contract_info.get("stepSize")))
        price_precision = int(contract_info.get("price") or contract_info.get("price_precision") or self._decimal_precision_from_step(contract_info.get("tickSize")))

        quantity_value = Decimal(str(order_data.get("quantity")))
        quantity_str = self._format_decimal(quantity_value, size_precision)
        quantity_units = int((self._quantize_decimal(quantity_value, size_precision) * (Decimal(10) ** underlying_decimals)).to_integral_value(rounding=ROUND_HALF_UP))
        if quantity_units <= 0:
            raise Exception("Order quantity is below Hibachi minimum step size")

        order_type = (order_data.get("type") or "MARKET").upper()
        side_flag = (order_data.get("side") or "BUY").upper()
        side_value = "BID" if side_flag == "BUY" else "ASK"
        side_code = 1 if side_value == "BID" else 0

        price_decimal = Decimal(str(order_data.get("price") if order_data.get("price") is not None else order_data.get("expected_price", 0)))
        price_for_signature = price_decimal if order_type == "LIMIT" else Decimal("0")
        price_units = int((price_for_signature * (Decimal(2) ** 32) * (Decimal(10) ** (underlying_decimals - settlement_decimals))).to_integral_value(rounding=ROUND_HALF_UP)) if price_for_signature else 0
        if order_type == "LIMIT" and price_units <= 0:
            raise Exception("Limit order price must be positive for Hibachi")

        max_fee_rate = Decimal(str(
            (self._fee_config or {}).get("tradeTakerFeeRate")
            or (self._fee_config or {}).get("tradeMakerFeeRate")
            or "0.0005"
        ))
        max_fee_rate = max_fee_rate.quantize(Decimal(1).scaleb(-8), rounding=ROUND_HALF_UP)
        max_fee_units = int((max_fee_rate * (Decimal(10) ** 8)).to_integral_value(rounding=ROUND_HALF_UP))
        max_fee_str = f"{max_fee_rate:.8f}".rstrip("0").rstrip(".") or "0"

        nonce = self._generate_nonce()

        payload = (
            self._pack_uint(nonce, 8)
            + self._pack_uint(contract_id, 4)
            + self._pack_uint(quantity_units, 8)
            + self._pack_uint(side_code, 4)
            + self._pack_uint(price_units, 8)
            + self._pack_uint(max_fee_units, 8)
        )

        signature = self._sign_payload(payload)

        account_id = self._account_id()
        if not account_id:
            raise Exception("Account identifier is required to place orders on Hibachi")

        body = {
            "accountId": int(account_id) if str(account_id).isdigit() else account_id,
            "symbol": normalized_symbol,
            "nonce": nonce,
            "side": side_value,
            "orderType": order_type,
            "quantity": quantity_str,
            "maxFeesPercent": max_fee_str,
            "signature": signature,
        }

        if order_type == "LIMIT" and order_data.get("price") is not None:
            body["price"] = self._format_decimal(price_decimal, price_precision)

        response = await self.send_request(
            method="POST",
            endpoint="order",
            data=body,
        )

        order_id = response.get("orderId") or response.get("order_id")
        if order_id is None:
            raise Exception(f"Unexpected create order response: {response}")

        token_name = order_data.get("token_name") or self._token_from_symbol(normalized_symbol)

        try:
            order_snapshot = await self.get_order_result(token_name or normalized_symbol, order_id)
        except Exception as exc:
            logger.opt(colors=True).warning(
                f"[!] <white>{self.label}</white> | Failed to fetch order snapshot after placement: {exc}"
            )
            fallback_price = price_decimal if price_decimal is not None else Decimal(str(order_data.get("expected_price") or "0"))
            executed_qty = quantity_value if order_type != "LIMIT" else Decimal("0")
            order_snapshot = {
                "orderId": order_id,
                "status": "FILLED" if order_type == "MARKET" else "NEW",
                "avgPrice": str(fallback_price),
                "executedQty": str(executed_qty),
                "cumQuote": str(fallback_price * executed_qty),
            }

        return order_snapshot

    async def get_balance(self):
        response = await self.send_request(
            method="GET",
            endpoint="balance",
        )
        balance_value = response.get("balance")
        try:
            return float(balance_value)
        except (TypeError, ValueError):
            raise Exception(f'Failed to get balance: {response}')

    async def get_token_price(self, token_name: str):
        response = await self.send_request(
            method="GET",
            endpoint="ticker_price",
            params={"symbol": f"{token_name}/USDT-P"},
        )
        price = response.get("markPrice") or response.get("tradePrice")
        if price is None:
            raise Exception(f'Failed to get {token_name} price: {response}')

        return Decimal(price)

    async def get_leverages(self):
        response = await self.send_request(
            method="GET",
            endpoint="account",
        )
        leverages = response.get("leverages")
        if leverages is None:
            raise Exception(f'Failed to get leverages: {response}')

        result: dict[str, int] = {}
        for entry in leverages:
            symbol = entry.get("symbol")
            rate = entry.get("initialMarginRate")
            if not symbol or rate is None:
                continue
            try:
                numeric_rate = float(rate)
                leverage_value = max(int(round(1 / numeric_rate))) if numeric_rate else 1
            except (TypeError, ValueError, ZeroDivisionError):
                leverage_value = 1
            result[symbol.replace("/USDT-P", "")] = max(leverage_value, 1)
        return result

    async def change_leverage(self, token_name: str, leverage: int):
        logger.opt(colors=True).debug(
            f"[•] <white>{self.label}</white> | Skipping leverage update for <white>{token_name}</white> (Hibachi API not available)"
        )

    async def get_account_positions(self):
        response = await self.send_request(
            method="GET",
            endpoint="position_risk",
        )
        positions = response.get("positions")
        if positions is None:
            raise Exception(f'Failed to get account positions: {response}')

        normalized = []
        for position in positions:
            quantity = Decimal(str(position.get("quantity", "0")))
            if not quantity:
                continue

            direction = str(position.get("direction", "")).strip().lower()
            if direction == "short":
                signed_quantity = -quantity
            elif direction == "long" or direction == "both":
                signed_quantity = quantity
            else:
                # HIBACHI-CHANGE: fallback to signed quantity field when direction is missing.
                signed_quantity = Decimal(str(position.get("signedQuantity", quantity)))
                if not signed_quantity:
                    signed_quantity = quantity

            symbol = position.get("symbol", "")
            normalized.append(
                {
                    "symbol": symbol.replace("/USDT-P", "USDT"),
                    "positionAmt": str(signed_quantity),
                }
            )
        return normalized

    async def get_account_orders(self):
        candidates = self._open_orders_candidates()
        params: dict | None = None

        account_id = self._account_id()
        if account_id:
            normalized = int(account_id) if str(account_id).isdigit() else account_id
            params = {"accountId": normalized}

        last_not_found: Exception | None = None

        for base, path in candidates:
            try:
                url = self._resolve_base_url(base, path)
            except ValueError as exc:
                logger.opt(colors=True).debug(
                    f"[•] <white>{self.label}</white> | Skipping invalid open-orders path <white>{path}</white>: {exc}"
                )
                continue

            try:
                response = await self.send_request(
                    method="GET",
                    endpoint=None,
                    url=url,
                    params=dict(params) if params else None,
                )
            except Exception as exc:
                message = str(exc)
                if "HTTP 404" in message or "Not Found" in message:
                    last_not_found = exc
                    continue
                raise

            if response is None:
                return []

            if isinstance(response, list):
                return response

            if isinstance(response, dict):
                for key in ("orders", "data", "items"):
                    value = response.get(key)
                    if isinstance(value, list):
                        return value
                    if value is None:
                        continue

                if not response:
                    return []

            logger.opt(colors=True).debug(
                f"[•] <white>{self.label}</white> | Unexpected open-orders payload from <white>{url}</white>: {response}"
            )

        if last_not_found is not None:
            raise last_not_found

        return []

    async def close_all_open_orders(self, token_name: str):
        await self._ensure_tokens_cached()
        account_id = self._account_id()
        if not account_id:
            raise Exception("Account identifier is required to cancel orders")

        nonce = self._generate_nonce()
        payload = self._pack_uint(nonce, 8)
        signature = self._sign_payload(payload)

        request_body = {
            "accountId": int(account_id) if str(account_id).isdigit() else account_id,
            "nonce": nonce,
            "signature": signature,
        }

        if token_name:
            contract = self._lookup_contract(token_name)
            if contract:
                request_body["contractId"] = int(contract.get("contract_id") or contract.get("id"))

        await self.send_request(
            method="DELETE",
            endpoint="cancel_all",
            data=request_body,
        )

    async def get_token_order_book(self, token_name: str):
        response = await self.send_request(
            method="GET",
            endpoint="order_book",
            params={"symbol": f"{token_name}/USDT-P", "depth": 5},
        )
        bid = response.get("bid")
        ask = response.get("ask")
        if not bid or not ask:
            raise Exception(f'Failed to get {token_name} order book: {response}')

        def _first_price(side: dict) -> float:
            levels = side.get("levels") or []
            if not levels:
                return 0.0
            return float(levels[0].get("price", 0))

        return {"BUY": _first_price(bid), "SELL": _first_price(ask)}

    async def get_order_result(self, token_name: str, order_id: int | str):
        await self._ensure_tokens_cached()

        params = {"orderId": order_id}
        response = await self.send_request(
            method="GET",
            endpoint="order",
            params=params,
        )
        if not isinstance(response, dict):
            raise Exception(f"Unexpected order response: {response}")

        raw_status = (response.get("status") or "").upper()
        if raw_status in {"PLACED", "OPEN"}:
            mapped_status = "NEW"
        elif raw_status in {"FILLED", "EXECUTED"}:
            mapped_status = "FILLED"
        elif raw_status in {"CANCELLED", "CANCELED"}:
            mapped_status = "CANCELED"
        else:
            mapped_status = raw_status or "UNKNOWN"

        contract = self._lookup_contract(token_name)
        size_precision = 0
        price_precision = 0
        if contract:
            size_precision = int(contract.get("size") or contract.get("size_precision") or self._decimal_precision_from_step(contract.get("stepSize")))
            price_precision = int(contract.get("price") or contract.get("price_precision") or self._decimal_precision_from_step(contract.get("tickSize")))

        total_qty = Decimal(str(response.get("totalQuantity") or response.get("quantity") or response.get("origQty") or "0"))
        available_qty = Decimal(str(response.get("availableQuantity") or response.get("leavesQuantity") or "0"))
        executed_qty = total_qty - available_qty
        if executed_qty < 0:
            executed_qty = Decimal("0")
        if mapped_status == "FILLED" and executed_qty == 0 and total_qty:
            executed_qty = total_qty

        price_field = (
            response.get("avgPrice")
            or response.get("price")
            or response.get("executionPrice")
            or response.get("lastPrice")
            or response.get("markPrice")
            or response.get("triggerPrice")
            or 0
        )
        price_decimal = Decimal(str(price_field or "0"))

        cum_quote = price_decimal * executed_qty

        executed_qty_str = self._format_decimal(executed_qty, size_precision) if contract else str(executed_qty.normalize())
        price_str = self._format_decimal(price_decimal, price_precision) if contract else str(price_decimal.normalize())
        cum_quote_str = str(cum_quote.normalize()) if cum_quote else "0"

        return {
            "orderId": response.get("orderId") or response.get("id") or str(order_id),
            "status": mapped_status,
            "avgPrice": price_str,
            "executedQty": executed_qty_str,
            "cumQuote": cum_quote_str,
        }

    async def cancel_order(self, token_name: str, order_id: int | str):
        await self._ensure_tokens_cached()
        account_id = self._account_id()
        if not account_id:
            raise Exception("Account identifier is required to cancel a specific order")

        contract = self._lookup_contract(token_name)
        contract_id = int(contract.get("contract_id") or contract.get("id")) if contract else None

        order_identifier = str(order_id)
        try:
            order_numeric = int(order_identifier)
            payload = self._pack_uint(order_numeric, 8)
            request_body = {
                "accountId": int(account_id) if str(account_id).isdigit() else account_id,
                "orderId": order_numeric,
                "signature": self._sign_payload(payload),
            }
        except ValueError:
            payload = order_identifier.encode("utf-8")
            request_body = {
                "accountId": int(account_id) if str(account_id).isdigit() else account_id,
                "clientId": order_identifier,
                "signature": self._sign_payload(payload),
            }

        if contract_id is not None:
            request_body["contractId"] = contract_id

        await self.send_request(
            method="DELETE",
            endpoint="order",
            data=request_body,
        )


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