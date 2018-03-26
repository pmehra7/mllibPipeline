package com.examples.ml


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession


object App {
  def main(args: Array[String]):Unit = {

    val spark = SparkSession
      .builder
      .appName("mllib Pipeline Application")
      .config("spark.cassandra.input.consistency.level","LOCAL_QUORUM")
      .enableHiveSupport()
      .getOrCreate()

    val housingData = spark.sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("dsefs:///data/housing.csv")

    val Array(training, test) = housingData.randomSplit(Array(0.8, 0.2), seed = 12345)

    val featureColumns = Array("longitude", "latitude", "housing_median_age", "total_rooms", "population", "households", "median_income", "total_bedrooms_imp", "ocean_proximity_index", "ocean_proximity_vector", "rooms_per_household", "population_per_household", "bedrooms_per_room")

    val testImputer = new Imputer()
      .setInputCols(Array("total_bedrooms"))
      .setOutputCols(Array("total_bedrooms_imp"))
      .setStrategy("median")

    val indexer = new StringIndexer()
      .setInputCol("ocean_proximity")
      .setOutputCol("ocean_proximity_index")

    val encoder = new OneHotEncoder()
      .setInputCol("ocean_proximity_index")
      .setOutputCol("ocean_proximity_vector")

    val populationPerHousehold = new DivisionTransformer("PopulationPerHoushold")
      .setInputColDividend("population")
      .setInputColDivisor("households")
      .setOutputCol("population_per_household")

    val roomsPerHousehold = new DivisionTransformer("RoomsPerHoushold")
      .setInputColDividend("total_rooms")
      .setInputColDivisor("households")
      .setOutputCol("rooms_per_household")

    val bedroomsPerHousehold = new DivisionTransformer("BedroomsPerRoom")
      .setInputColDividend("total_bedrooms_imp")
      .setInputColDivisor("total_rooms")
      .setOutputCol("bedrooms_per_room")

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    val getFeature = new RelevantDataTransformer("setFeaturesAndLabel")
      .setInputColFeatures("scaledFeatures")
      .setInputColLabel("median_house_value")

    val lr = new LinearRegression()
    val pipeline_lr = new Pipeline().setStages(Array(testImputer,indexer, encoder, roomsPerHousehold, bedroomsPerHousehold, populationPerHousehold, assembler, scaler, getFeature, lr))

    val paramGrid_lr = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(.4, .8))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.maxIter, Array(10, 30, 60))
      .build()

    val cvalidate_lr = new CrossValidator()
      .setEstimator(pipeline_lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid_lr)
      .setNumFolds(5)

    var modelLR = cvalidate_lr.fit(training)
    var predictionsLR = modelLR.transform(test)
    val metricsLR = new ModelEvaluatorRDD(predictionsLR)
      .setEvalType("linear")
    print(metricsLR.modelMetrics)


    val dt = new DecisionTreeRegressor()
    val pipeline_dr = new Pipeline().setStages(Array(testImputer,indexer, encoder, roomsPerHousehold, bedroomsPerHousehold, populationPerHousehold, assembler, scaler, getFeature, dt))

    val paramGrid_dr = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(10, 15, 20, 25))
      .addGrid(dt.maxBins, Array(5, 10, 15))
      .build()

    val cvalidate_dr = new CrossValidator()
      .setEstimator(pipeline_dr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid_dr)
      .setNumFolds(5)

    val modelDT = cvalidate_dr.fit(training)
    val predictionsDR = modelDT.transform(test)
    val metricsDR = new ModelEvaluatorRDD(predictionsDR)
      .setEvalType("linear")
    print(metricsDR.modelMetrics)

    // Persist Pipeline:
    //cvalidate.write.overwrite().save("/home/examples/mlPipeline")
    //val myStoredModel = CrossValidatorModel.load("/home/examples/mlPipeline")

  }
}
