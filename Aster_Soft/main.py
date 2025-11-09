from os import name as os_name
from random import randint
from loguru import logger
from time import sleep
import asyncio

from modules.retry import DataBaseError
from modules.utils import choose_mode, async_sleep
from modules import *
import settings


def initialize_account(module_data: dict, db: DataBase, group_data: dict = None):
    lock_identifier = module_data.get("lock_identifier") or module_data.get("address")
    account_reference = module_data.get("account_reference")  # HIBACHI-CHANGE: decoded Hibachi account id
    wallet_address = module_data.get("wallet_address")  # HIBACHI-CHANGE: decoded Hibachi wallet address
    browser = Browser(
        proxy=module_data["proxy"],
        api_key=module_data["apikey"],
        label=module_data["label"],
        account_reference=account_reference,
    )
    aster_client = AsterClient(
        browser=browser,
        apikey=module_data["apikey"],
        encoded_apikey=module_data["encoded_apikey"],
        address=module_data["address"],
        lock_identifier=lock_identifier,
        account_reference=account_reference,  # HIBACHI-CHANGE
        wallet_address=wallet_address,  # HIBACHI-CHANGE
        label=module_data["label"],
        proxy=module_data["proxy"],
        db=db,
        group_data=group_data,
    )

    return aster_client


async def run_modules(mode: int, module_data: dict, sem: asyncio.Semaphore, db: DataBase):
    lock_id = module_data.get("lock_identifier", module_data["address"])
    async with address_locks[lock_id]:
        async with sem:
            aster_client = None
            try:
                aster_client = initialize_account(module_data, db=db)
                module_data["module_info"]["status"] = await aster_client.run_mode(mode=mode)

            except Exception as err:
                if aster_client:
                    logger.error(f'[-] {aster_client.label} | Account Error: {err}')
                    await db.append_report(key=aster_client.encoded_apikey, text=str(err), success=False)
                else:
                    logger.error(f'[-] {module_data["label"]} | Global error: {err}')

            finally:
                if aster_client:
                    await aster_client.browser.close_session()  # ВИПРАВЛЕНО: close_session замість close_sessions

                if type(module_data) == dict:
                    if mode in [1, 2]:
                        last_module = await db.remove_module(module_data=module_data)
                    else:
                        last_module = await db.remove_account(module_data=module_data)

                    # ВИПРАВЛЕННЯ: перевірка чи aster_client існує перед використанням
                    encoded_apikey = aster_client.encoded_apikey if aster_client else module_data.get("encoded_apikey", "unknown")
                    label = aster_client.label if aster_client else module_data.get("label", "unknown")
                    address = aster_client.address if aster_client else module_data.get("address", "unknown")

                    reports = await db.get_account_reports(
                        key=encoded_apikey,
                        label=label,
                        address=address,
                        last_module=last_module,
                        mode=mode,
                    )
                    await TgReport().send_log(logs=reports)

                    if module_data["module_info"]["status"] is True:
                        await async_sleep(randint(*settings.SLEEP_AFTER_ACC))
                    else:
                        await async_sleep(10)


async def run_pair(mode: int, group_data: dict, sem: asyncio.Semaphore, db: DataBase):
    async with MultiLock([
        wallet_data.get("lock_identifier", wallet_data["address"]) for wallet_data in group_data["wallets_data"]
    ]):
        async with sem:
            aster_clients = []
            try:
                aster_clients = [
                    initialize_account(wallet_data, db=db, group_data=group_data)
                    for wallet_data in group_data["wallets_data"]
                ]
                group_data["module_info"]["status"] = await PairAccounts(
                    accounts=aster_clients,
                    group_data=group_data
                ).run(mode=mode)

            except Exception as err:
                logger.error(f'[-] Group {group_data["group_number"]} | Global error | {err}')
                await db.append_report(key=group_data["group_index"], text=str(err), success=False)

            finally:
                if aster_clients:
                    for aster_client in aster_clients:
                        await aster_client.browser.close_session()  # ВИПРАВЛЕНО: close_session замість close_sessions
                await db.remove_group(group_data=group_data)

                reports = await db.get_account_reports(
                    key=group_data["group_index"],
                    label=f"Group {group_data['group_number']}",
                    address=None,
                    last_module=False,
                    mode=mode,
                )
                await TgReport().send_log(logs=reports)

                if group_data["module_info"]["status"] is True:
                    to_sleep = randint(*settings.SLEEP_AFTER_ACC)
                    logger.opt(colors=True).debug(f'[•] <white>Group {group_data["group_number"]}</white> | Sleep {to_sleep}s')
                    await async_sleep(to_sleep)
                else:
                    await async_sleep(10)


async def runner(mode: int, db: DataBase):
    sem = asyncio.Semaphore(settings.THREADS)

    if mode in [3]:
        all_groups = db.get_all_groups()
        if all_groups != 'No more accounts left':
            await asyncio.gather(*[
                run_pair(group_data=group_data, mode=mode, sem=sem, db=db)
                for group_data in all_groups
            ])

    else:
        all_modules = db.get_all_modules(unique_wallets=mode in [4, 5])
        if all_modules != 'No more accounts left':
            await asyncio.gather(*[
                run_modules(module_data=module_data, mode=mode, sem=sem, db=db)
                for module_data in all_modules
            ])

    logger.success(f'All accounts done.')
    return 'Ended'


if __name__ == '__main__':
    if os_name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        db = DataBase()

        while True:
            mode = choose_mode()

            match mode.type:
                case "database":
                    db.create_modules(mode=mode.soft_id)

                case "module":
                    if asyncio.run(runner(mode=mode.soft_id, db=db)) == 'Ended': break
                    print('')

        sleep(0.1)
        input('\n > Exit\n')

    except DataBaseError as e:
        logger.error(f'[-] Database | {e}')

    except KeyboardInterrupt:
        pass

    finally:
        logger.info('[•] Soft | Closed')