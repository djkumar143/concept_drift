from datetime import datetime
import pytz
import logging
from typing import Tuple
import time

import psycopg2
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

POSTGRES_URL = "jdbc:postgresql://postgres:5432/streaming_db"

india_tz = pytz.timezone("Asia/Kolkata")

POSTGRES_PROPERTIES = {
    "user": "postgres",
    "password": "postgres",
    "driver": "org.postgresql.Driver"
}

POSTGRES_CONFIG = {
    "host": "postgres",
    "database": "streaming_db",
    "user": "postgres",
    "password": "postgres"
}

TABLE_NAME = "market_features"

# store predictions in the table
def write_predictions(df: DataFrame) -> None:
    df.write.mode("append").jdbc(
        url=POSTGRES_URL,
        table=TABLE_NAME,
        properties=POSTGRES_PROPERTIES
    )

# update true_label in the table
def update_true_label(
    eventID: str,
    true_label: str,
    arrival_time: datetime
) -> None:

    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE market_features
                SET
                    true_label = %s,
                    label_arrival_time = %s
                WHERE
                    eventID = %s
                """,
                (
                    true_label,
                    arrival_time,
                    eventID
                )
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"No prediction found for eventID={eventID}"
                )

            conn.commit()

            logger.info(
                "Updated true label for eventID=%s",
                eventID
            )
            
# get predicted value from the table           
def get_prediction(eventID:str)-> Tuple[str, str]:
    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    predicted_label,
                    model_version
                FROM market_features
                WHERE eventID = %s
                """,
                (
                    eventID,
                )
            )
            # fetchone(): returns a tuple, only one row at a time
            # fetchall(): returns a list of tuples
            row = cursor.fetchone()
            
            if row is None:
                return None

            return{
                "predicted_label" : row[0],
                "model_version" : row[1]
            }
            
def wait_for_prediction():
    while True:
        with psycopg2.connect(**POSTGRES_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM
                    market_features
                    """
                )
                count = cursor.fetchone()[0]
        if count > 0:
            logger.info("Prediction found. Starting performance monitor.")
            break
        logger.info("Waiting for prediction.")
        time.sleep(2)

            
# get retraining status
def get_retraining_status()-> bool:
    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT running
                FROM retraining_status
                WHERE id = 1
                """
            )
            row = cursor.fetchone()
            
            if row is None:
                return False
            return row[0]
            

# to start retraining the model
def start_retraining()-> None:
    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retraining_status
                SET 
                    running = TRUE,
                    started_at = %s
                WHERE id = 1
                """,
                (
                    datetime.now(india_tz),
                )
            )
            conn.commit()

def finish_retraining()-> None:
    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retraining_status
                SET
                    running = FALSE,
                    started_at = NULL
                WHERE id = 1
                """
            )
            conn.commit()