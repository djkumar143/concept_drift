from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression #estimator(used for training)
from utils.preprocessing import prepare_features
from pyspark.sql.functions import when, col, lit
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("train_base_model") \
    .master("local[*]") \
    .getOrCreate()
    
data_path = "/app/data/elec2_past_data.csv"
df = spark.read.csv(
    data_path,
    inferSchema=True,
    header=True)

df = df.select(
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
    "class"
)


df = df.withColumn(
    "label", when(col("class") == "DOWN", lit(0)) \
        .otherwise(lit(1)).cast(IntegerType())
)

training_data = prepare_features(df, label_col="label")

train_df, test_df = training_data.randomSplit(
    weights=[0.8,0.2],
    seed=42
)


lr = LogisticRegression(
    featuresCol= "features",
    labelCol="label"
)
model = lr.fit(train_df)

predictions = model.transform(test_df)


evaluator = MulticlassClassificationEvaluator(
    predictionCol= "prediction",
    labelCol= "label",
    metricName= "accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

model.write().overwrite().save("/app/models/lr_v1")

spark.stop()