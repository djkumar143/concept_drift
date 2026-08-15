# Weather the distribution of a feature has changed or not
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from utils.kswin_utils import KSWINDetector

def create_spark_session():
    return SparkSession.builder \
        .appName("feature_monitor") \
        .getOrCreate()

#indivisual detectors for each feature
detectors = {
    "nswprice": KSWINDetector(),
    "nswdemand": KSWINDetector(),
    "vicprice": KSWINDetector(),
    "vicdemand": KSWINDetector(),
    "transfer": KSWINDetector()
}

def process_batch(batch_df, batch_id):
    rows = batch_df.select(
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer"
    ).collect() #records from all the driver nodes accumulated in a list at driver node
    
    #use if want to evaluate all features together
    feature_results = {}
    
    for row in rows:
        for feature, detector in detectors.items():
            if not detector.is_ready():
                print(
                    f"{feature}: "
                    f"{detector.current_size()}/{detector.window_size}"
                )
            #get the kswin results
            result = detector.update(row[feature])
            feature_results[feature] = result
            
            if detector.is_ready():
                status = "DRIFT" if result["drift"] else "NO DRIFT"

                print(
                    f"{feature}: {status} ",
                    f"KS={result['ks_statistic']:.4f}",
                    f"(p={result['p_value']:.6f})"
                )

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
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
        .option("subscribe", topic) \
        .load()
    
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING) AS value") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")
        
    def batch_processor(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        process_batch(batch_df, batch_id)
            
    query = parsed_df.writeStream \
        .foreachBatch(batch_processor) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/checkpoints/feature_monitor") \
        .queryName("feature_monitor") \
        .start()
    
    query.awaitTermination()

if __name__ == "__main__":
    main()