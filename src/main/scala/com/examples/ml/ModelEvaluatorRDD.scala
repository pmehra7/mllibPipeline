package com.examples.ml

import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.sql.DataFrame

// In progress: get model metrics from a dataframe, currently only RDD
class ModelEvaluatorRDD(val predictionsDF: DataFrame, var evalType: String) {

  // Default Values
  var predictionCol: String = "prediction"
  var labelCol: String = "label"
  val modelMetrics = modelType(evalType)

  def this(predictionsDF: DataFrame) {
    this(predictionsDF, "linear")
  }

  def setEvalType(evalType: String) = {
    this.evalType = validateEvaluator(evalType)
    this
  }
  def getEvalType() = {
    this.evalType
  }
  def setPredictionCol(predictionCol: String) = {
    this.predictionCol = predictionCol
    this
  }
  def getPredictionCol() = {
    this.predictionCol
  }
  def setLabelCol(labelCol: String) = {
    this.labelCol = labelCol
    this
  }
  def getLabelCol() = {
    this.labelCol
  }

  // Keep private unless there is a reason to expose
  lazy private val predictions = predictionsDF.select(predictionCol).rdd.map(_.getDouble(0))
  lazy private val labels = predictionsDF.select(labelCol).rdd.map(_.getDouble(0))

  // Regression
  private class Regression {
    val metrics = new RegressionMetrics(predictions.zip(labels))
    val RMSE: Double = metrics.rootMeanSquaredError
    val R2: Double = metrics.r2
    val expVar: Double = metrics.explainedVariance
    override def toString = s"RMSE: $RMSE \n R2: $R2 \n Exp. Var: $expVar"
  }

  // Multi Classification
  private class MultiClassification {
    val metrics = new MulticlassMetrics(predictions.zip(labels))
    val accuracy: Double = metrics.accuracy
    val weightedTruePositiveRate: Double = metrics.weightedTruePositiveRate
    override def toString = s"Accuracy: $accuracy \n Weighted True Positive: $weightedTruePositiveRate"
  }

  // Do these later
  // BinaryClassification
  private class BinaryClassification { }
  // MultiLabel
  private class MultiLabel { }
  // Ranking
  private class Ranking { }

  // Helper Functions used to select the correct Model Metrics Class
  private def validateEvaluator(s: String) = {
    val evalTypes: List[String] = List("linear", "multiclass", "binaryclass", "multilabel", "ranking")
    if (evalTypes.contains(s.toLowerCase)) s.toLowerCase
    else throw new IllegalArgumentException("Choose either: linear, multiclass, binaryclass, multilablel, or ranking")
  }

  private def modelType(s: String) = {
    var modelMetrics = s match {
      case "linear" => new Regression()
      case "multiclass" => new MultiClassification()
      case "binaryclass" => new BinaryClassification()
      case "multilabel" => new MultiLabel()
      case "ranking" => new Ranking()
      case _ => throw new Exception("Error Selecting Correct Metrics Class for Given Model")
    }
    modelMetrics
  }

}
