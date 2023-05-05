# -*- coding: utf-8 -*-
import click
import logging
import pyarrow.feather as feather
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(input_filepath)
    id_column = next((col for col in df.columns if col.lower() == 'id'), None)
    if id_column:
        # Set the index to the ID column
        df.set_index(id_column, inplace=True)
    else:
        print("No 'id', 'Id', or 'ID' column found.")
    print(df.info())
    print(df)
    feather.write_feather(df, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
