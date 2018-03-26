package com.examples.ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}



class RelevantDataTransformer(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("RelevantDataTransformer"))

  final val inputColFeatures: Param[String] = new Param[String](this, "inputColFeatures", "Input column name - Features")

  final def getInputColFeatures: String = $(inputColFeatures)

  final def setInputColFeatures(value: String): RelevantDataTransformer = set(inputColFeatures, value)

  final val inputColLabel: Param[String] = new Param[String](this, "inputColLabel", "Input column name Label")

  final def getInputColLabel: String = $(inputColLabel)

  final def setInputColLabel(value: String): RelevantDataTransformer = set(inputColLabel, value)

  override def transform(dataset: Dataset[_]): DataFrame = {

    dataset.select(col($(inputColFeatures)) as "features", col($(inputColLabel)) as "label")

  }

  override def transformSchema(schema: StructType): StructType = {

    val featureDataType = schema($(inputColFeatures)).dataType
    require(featureDataType.equals(VectorType),
      s"Column ${$(inputColFeatures)} must be a Vector but was actually $featureDataType.")

    val labelDataType = schema($(inputColLabel)).dataType
    require(labelDataType.equals(DataTypes.DoubleType),
      s"Column ${$(inputColLabel)} must be DoubleType but was actually $labelDataType.")

    schema.add(StructField("features", VectorType, false))
    schema.add(StructField("label", DataTypes.DoubleType, false))

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}