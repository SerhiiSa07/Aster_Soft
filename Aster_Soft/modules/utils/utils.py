from http.cookies import SimpleCookie
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from random import randint
from typing import Optional, Tuple

from loguru import logger
from time import sleep
from web3 import Web3
from tqdm import tqdm
import asyncio
import string
import sys
sys.__stdout__ = sys.stdout # error with `import inquirer` without this string in some system


logger.remove()
logger.add(sys.stderr, format="<white>{time:HH:mm:ss}</white> | <level>{message}</level>")


def sleeping(*timing):
    if type(timing[0]) == list: timing = timing[0]
    if len(timing) == 2: x = randint(timing[0], timing[1])
    else: x = timing[0]
    desc = datetime.now().strftime('%H:%M:%S')
    if x <= 0: return
    for _ in tqdm(range(x), desc=desc, bar_format='{desc} | [•] Sleeping {n_fmt}/{total_fmt}'):
        sleep(1)


def make_border(
        table_elements: dict,
        keys_color: str | None = None,
        values_color: str | None = None,
        table_color: str | None = None,
):
    def tag_color(value: str, color: str | None):
        if keys_color:
            return f"<{color}>{value}</{color}>"
        return value

    left_margin = 35
    space = 2
    horiz = '━'
    vert = '║'
    conn = 'o'

    if not table_elements: return "No text"

    key_len = max([len(key) for key in table_elements.keys()])
    val_len = max([len(str(value)) for value in table_elements.values()])
    text = f'{" " * left_margin}{conn}{horiz * space}'

    text += horiz * (key_len + space) + conn
    text += horiz * space
    text += horiz * (val_len + space) + conn

    text += '\n'

    for table_index, element in enumerate(table_elements):
        text += f'{" " * left_margin}{vert}{" " * space}'

        text += f'{tag_color(element, keys_color)}{" " * (key_len - len(element) + space)}{vert}{" " * space}'
        text += f'{tag_color(table_elements[element], values_color)}{" " * (val_len - len(str(table_elements[element])) + space)}{vert}'
        text += "\n" + " " * left_margin + conn + horiz * space
        text += horiz * (key_len + space) + conn
        text += horiz * (space * 2 + val_len) + conn + '\n'
    return tag_color(text, table_color)


def format_password(password: str):
    # ADD UPPER CASE
    if not any([password_symbol in string.ascii_uppercase for password_symbol in password]):
        first_letter = next(
            (symbol for symbol in password if symbol in string.ascii_letters),
            "i"
        )
        password += first_letter.upper()

    # add lower case
    if not any([password_symbol in string.ascii_lowercase for password_symbol in password]):
        first_letter = next(
            (symbol for symbol in password if symbol in string.ascii_letters),
            "f"
        )
        password += first_letter.lower()

    # add numb3r5
    if not any([password_symbol in string.digits for password_symbol in password]):
        password += str(len(password))[0]

    # add $ymbol$
    symbols_list = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'
    if not any([password_symbol in symbols_list for password_symbol in password]):
        password += symbols_list[sum(ord(c) for c in password) % len(symbols_list)]

    # add 8 characters
    if len(password) < 8:
        all_symbols = string.digits + string.ascii_letters
        password += ''.join(
            all_symbols[sum(ord(c) for c in password[:i+1]) % len(symbols_list)]
            for i in range(max(0, 8 - len(password)))
        )

    return password


def get_address(pk: str):
    return Web3().eth.account.from_key(pk).address


def parse_cookies(cookies: str, key: str):
    cookie = SimpleCookie()
    cookie.load(cookies)
    return cookie[key].value if cookie.get(key) else None


def get_response_error_reason(response: dict):
    return str(response.get("errors", [{}])[0].get("message", response)).removeprefix("Authorization: ")


def round_cut(value: float | str | Decimal, digits: int):
    return Decimal(str(int(Decimal(str(value)) * 10 ** digits) / 10 ** digits))


async def async_sleep(seconds: int):
    for _ in range(int(seconds)):
        await asyncio.sleep(1)


@dataclass(frozen=True)
class ParsedAPIKey:
    label: str
    credential: str
    credential_parts: Tuple[str, ...]
    account_id: Optional[str]
    api_key: str
    secret_key: str
    wallet_private_key: Optional[str]
    wallet_address: Optional[str]

    @property
    def lock_identifier(self) -> str:
        if self.wallet_address:
            return self.wallet_address
        if self.account_id:
            return self.account_id
        return self.credential_parts[0]


def parse_api_key(raw_key: str, default_label: str) -> ParsedAPIKey:
    cleaned_key = raw_key.strip()
    if not cleaned_key:
        raise ValueError("API key entry cannot be empty")

    parts = cleaned_key.split(":")

    if parts[-1].startswith("0x"):
        if len(parts) < 3:
            raise ValueError(f"Unexpected API key format: {cleaned_key}")
        credential_parts = tuple(parts[-3:])
        label_parts = parts[:-3]
    else:
        if len(parts) < 2:
            raise ValueError(f"Unexpected API key format: {cleaned_key}")
        credential_parts = tuple(parts[-2:])
        label_parts = parts[:-2]

    # HIBACHI-FORMAT: treat trailing numeric prefixes as part of the account identifier
    numeric_prefix: tuple[str, ...] = ()
    if label_parts:
        split_point = len(label_parts)
        while split_point > 0 and label_parts[split_point - 1].isdigit():
            split_point -= 1
        if split_point != len(label_parts):
            numeric_prefix = tuple(label_parts[split_point:])
            label_parts = label_parts[:split_point]

    if any(part == "" for part in credential_parts):
        raise ValueError(f"Unexpected API key format: {cleaned_key}")

    label = ":".join(label_parts).strip() if label_parts else default_label
    if not label:
        label = default_label

    account_id: Optional[str] = None
    if len(credential_parts) > 2:
        account_id_segments = list(numeric_prefix)
        if account_id_segments and account_id_segments[-1] == credential_parts[0]:
            account_id_segments = account_id_segments[:-1]
        account_id_segments.append(credential_parts[0])
        account_id = ":".join(account_id_segments)

    api_key = credential_parts[-2]
    secret_key = credential_parts[-1]

    wallet_private_key = secret_key if secret_key.startswith("0x") else None
    wallet_address: Optional[str] = None
    if wallet_private_key:
        try:
            wallet_address = get_address(wallet_private_key)
        except Exception:
            wallet_address = None

    credential = ":".join(credential_parts)

    return ParsedAPIKey(
        label=label,
        credential=credential,
        credential_parts=credential_parts,
        account_id=account_id,
        api_key=api_key,
        secret_key=secret_key,
        wallet_private_key=wallet_private_key,
        wallet_address=wallet_address,
    )
