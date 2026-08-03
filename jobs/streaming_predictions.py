import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, DoubleType, TimestampType
from pyspark.sql.functions import from_json, col, current_timestamp, when, lit

from utils.preprocessing import prepare_features
from utils.model_utils import load_current_model, get_current_model_version
from utils.database import write_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():    
    return SparkSession.builder \
        .appName("Streaming_prediction") \
        .config("spark.sql.session.timeZone", "Asia/Kolkata") \
        .getOrCreate()


def create_prediction_dataframe(batch_df):
    prediction_features = prepare_features(batch_df)
    model = load_current_model()
    model_version = get_current_model_version()
    prediction_df = model.transform(prediction_features)
    output_df = (
        batch_df.join(
            prediction_df.select(
                "eventID",
                "prediction"
            )
            ,on="eventID"
            
        ).withColumn(
            "predicted_label",
            when(col("prediction")== 0, lit("DOWN"))
            .otherwise(lit("UP"))
            
        ).withColumn(
            "prediction_time",
            current_timestamp()
            
        ).withColumn(
            "model_version",
            lit(model_version)
            
        ).withColumn(
            "true_label",
            lit(None).cast(StringType())
            
        ).withColumn(
            "label_arrival_time",
            lit(None).cast(TimestampType())
            
        ).drop("prediction")
    )
    output_df = output_df.select(
        "eventID",
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
        "predicted_label",
        "true_label",
        "prediction_time",
        "label_arrival_time",
        "model_version"
    )
    return output_df


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark Session initialized.")
    logger.info(f"Spark version:{spark.version}")

    schema = StructType([
        StructField("eventID", StringType(), False),
        StructField("nswprice", DoubleType(), True),
        StructField("nswdemand", DoubleType(), True),
        StructField("vicprice", DoubleType(), True),
        StructField("vicdemand", DoubleType(), True),
        StructField("transfer", DoubleType(), True)
    ])
    
    topic = "market-features"
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka1:29092, kafka2:29092, kafka3:29092") \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 1) \
        .option("subscribe", topic) \
        .load()

    parsed_df = kafka_df.selectExpr("CAST(value AS STRING) AS value") \
        .select(
            from_json(col("value"), schema).alias("data")) \
        .select(
            "data.*" #selecting all the fields of the StructType column(data)
        )
        
    def predict_and_store(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        output_df = create_prediction_dataframe(batch_df)
        
        write_predictions(output_df)
        
    query = parsed_df.writeStream \
        .foreachBatch(predict_and_store) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/checkpoints/streaming_prediction") \
        .queryName("streaming_prediction") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()