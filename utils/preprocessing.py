from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col


# from spark DataFrame, prepare feature vector and return along with either label or eventID
def prepare_features(df, label_col = None):
    assembler = VectorAssembler(
        inputCols=[
            "nswprice",
            "nswdemand",
            "vicprice",
            "vicdemand",
            "transfer"
            ],
        outputCol="features"
    )
    transformed_df = assembler.transform(df)
    
    if label_col is not None:
        transformed_df = transformed_df.withColumn(
            "label",
            col(label_col)
        )
        return transformed_df.select("features", "label")
    
    
    return transformed_df.select("eventID", "features")