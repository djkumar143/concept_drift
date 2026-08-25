import logging
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import LogisticRegression

from utils.preprocessing import prepare_features
from utils.database import finish_retraining

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSTGRES_URL = "jdbc:postgresql://postgres:5432/streaming_db"

POSTGRES_PROPERTIES = {
    "user": "postgres",
    "password": "postgres",
    "driver": "org.postgresql.Driver"
}

def create_spark_session():
    return SparkSession.builder \
        .appName("retrain_model") \
        .getOrCreate()

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("Spark session initialized.")
    
    df = spark.read \
        .jdbc(
            url = POSTGRES_URL,
            table = "market_data",
            properties = POSTGRES_PROPERTIES
        )
        
    df = df.filter(
        col("true_label").isNotNull()
    )
    
    # retrain only if atleast 2000 true labels available
    if df.count() < 2000:
        logger.info("Not enough labelled data for retraining.")
        finish_retraining()
        spark.stop()
        return
    df = df.orderBy(col("label_arrival_time").desc()).limit(2000)
    
    df = df.select(
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
        "true_label"
    )
    df = df.withColumn(
        "true_label",
        when(col("true_label") == "UP", lit(1)) \
            .otherwise(lit(0)).cast(IntegerType())
    )
    
    training_df = prepare_features(
        df,
        label_col= "true_label"
    )
    
    lr = LogisticRegression(
        featuresCol = "features",
        labelCol="label"
    )
    
    try:
        #train model
        model = lr.fit(training_df)
        
        with open("/app/model_registry.json","r") as file:
            model_info = json.load(file)
            
        retraining_model_version = model_info["version"] + 1
        model_name = f"lr_v{retraining_model_version}"
        
        #save model
        model.write().overwrite().save(f"/app/models/{model_name}")
        new_model = {
            "current_version" : model_name,
            "version": retraining_model_version
        }
        #update model_registry
        with open("/app/model_registry.json", "w") as file:
            json.dump(new_model,file, indent = 4)
        logger.info("New model successfully trained and activated.")
        
    except Exception:
        logger.exception("Retraining Failed.")
        
    finally:
        finish_retraining()
        spark.stop()
    
if __name__ == "__main__":
    main()