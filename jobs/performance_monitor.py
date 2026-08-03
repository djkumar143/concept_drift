from datetime import datetime
import pytz
import time
import logging
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, from_json

from utils.database import wait_for_prediction, update_true_label, get_prediction, get_retraining_status, start_retraining
from utils.eddm_utils import EDDMDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

india_tz = pytz.timezone("Asia/Kolkata")

eddm = EDDMDetector()

wait_for_prediction()

def create_spark_session():
    return SparkSession.builder \
        .appName("performance_monitor") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()
        
def process_batch(batch_df, batch_id):
    rows = batch_df.collect()
    for row in rows:
        eventID = row["eventID"]
        true_label = row["label"]
        prediction = get_prediction(eventID)
        
        if prediction is None:
            print(f"No prediction found for eventID:{eventID}")
            continue
        
        predicted_label = prediction["predicted_label"]
        
        update_true_label(eventID, true_label, datetime.now(india_tz))
        
        is_correct = (predicted_label == true_label)
        
        result = eddm.update(is_correct)
        print(result)
        
        if result["drift"]:
            if get_retraining_status():
                print("Retraining already in progress.")
            else:
                start_retraining()
                
                subprocess.Popen([
                    "/opt/spark/bin/spark-submit",
                    "--conf",
                    "spark.jars.ivy=/tmp/.ivy2",
                    "--packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6,org.postgresql:postgresql:42.7.4",
                    "/app/jobs/retrain_model.py"
                ])
        
def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("Spark Session initialized.")
    logger.info(f"Spark version:{spark.version}")
    
    schema = StructType([
        StructField("eventID", StringType(), False),
        StructField("label", StringType(), True)
    ])
    
    topic = "market-labels"
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka1:29092, kafka2:29092, kafka3:29092") \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 1) \
        .option("subscribe", topic) \
        .load()
        
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING) AS value") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")
        
    def batch_processor(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        process_batch(batch_df, batch_id)
        time.sleep(1)
    
    query = parsed_df.writeStream \
        .foreachBatch(batch_processor) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/checkpoints/performance_monitor") \
        .queryName("performance_monitor") \
        .start()
    
    query.awaitTermination()

if __name__ == "__main__":
    main()