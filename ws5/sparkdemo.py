import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName("ws5-regression").getOrCreate()

input_path = sys.argv[1]

df = spark.read.csv(input_path, header=True, inferSchema=True)

print("Showing input data:")
df.show(5)

assembler = VectorAssembler(
    inputCols=["total_bill", "size"],
    outputCol="features"
)

lr = LinearRegression(
    featuresCol="features",
    labelCol="tip"
)

pipeline = Pipeline(stages=[assembler, lr])

train, test = df.randomSplit([0.8, 0.2], seed=131)

pipeline_model = pipeline.fit(train)

predictions = pipeline_model.transform(test)

print("Showing predictions:")
predictions.select("total_bill", "size", "tip", "prediction").show(10)

evaluator = RegressionEvaluator(
    labelCol="tip",
    predictionCol="prediction"
)

rmse = evaluator.setMetricName("rmse").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

lr_model = pipeline_model.stages[-1]

print("===== MODEL RESULTS =====")
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

spark.stop()
