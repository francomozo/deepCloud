import logging
from pathlib import Path

from crontab import CronTab
import yaml

from cron_config.configure_crontab import configure_crontab

def start_main():
    # Load confiuration
    stream = open("config.yaml", "r")
    dcfg = yaml.load(stream, yaml.FullLoader)  # dict

    # verbose = dcfg['crontab']['verbose']
    loglevel = dcfg["logging"]["loglevel"]

    # logging
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(
        filename="logs/admin.log",
        format="%(levelname)s:%(asctime)s: %(message)s",
        level=numeric_level,
    )
    logging.info("Start start.py")

    # Show config if verbose is True
    # if verbose:
    #     logging.debug('verbose is activated')
    #     print('----------------------')
    #     print('CONFIGURATION DICT:')
    #     for key, value in dcfg.items():
    #         print (key + " : " + str(value))
    #     print('----------------------')


    # Create directories
    (Path(dcfg["paths"]["psat"]) / "meta").mkdir(
        parents=True, exist_ok=True
    )
    logging.debug("Create psat folder")

    # (Path(dcfg['paths']['dataset']) / 'meta').mkdir(parents=True,
    #   exist_ok=True)
    (Path(dcfg["paths"]["history"])).mkdir(parents=True, exist_ok=True)
    logging.debug("Create history folder")

    configure_crontab(dcfg)  # see definition

    logging.info("Finish start.py")


if __name__ == '__main__':
    start_main()

