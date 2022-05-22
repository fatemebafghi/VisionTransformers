
import os
import click
import logzero
import json
import codecs
from logzero import logger
from models.Classifier import Model as cl_model
from models.Detector import Model as dt_model

click.command()
@click.option('-c', '--configfile', type=click.Path(exists=True, readable=True),
              help='config file for create pipeline in json format')


def main(configfile):
    """pass"""

    if not os.path.exists('logs/pylogs'):
        os.makedirs('logs/pylogs')
    logzero.logfile("logs/pylogs/logfile.log", maxBytes=1e6)

    with codecs.open(configfile, 'r', 'utf8') as f:
        jsonconfig = json.load(f)

    general_setting = jsonconfig.get('general_setting', {})

    mode = general_setting.get("mode","")

    if mode == "classify":
        model = cl_model()
        image_path = "index.jpeg"
        result = model.predict(image_path)
        logger.info(result)

    elif mode == "detect":
        pass

    else:
        logger.error("command not found")


if __name__ == '__main__':
    main()
