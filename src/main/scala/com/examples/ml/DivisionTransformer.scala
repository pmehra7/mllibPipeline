package com.examples.ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}


class DivisionTransformer(override val uid: String) extends Transformer  {

  def this() = this(Identifiable.randomUID("DivisionTransformer"))

  final val inputColDividend: Param[String] = new Param[String](this, "inputColDividend", "Input column name - Dividend")

  final def getInputColDividend: String = $(inputColDividend)

  final def setInputColDividend(value: String): DivisionTransformer = set(inputColDividend, value)

  final val inputColDivisor: Param[String] = new Param[String](this, "inputColDivisor", "Input column name - Divisor")

  final def getInputColDivisor: String = $(inputColDivisor)

  final def setInputColDivisor(value: String): DivisionTransformer = set(inputColDivisor, value)

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  final def getOutputCol: String = $(outputCol)

  final def setOutputCol(value: String): DivisionTransformer = set(outputCol, value)

  private val divisionFunction: ((Double, Double) => Double) = (arg0: Double, arg1: Double) => {arg0 / arg1}

  override def transform(dataset: Dataset[_]): DataFrame = {
    val divisionUdf = udf(divisionFunction)
    dataset.select(col("*"), divisionUdf(col($(inputColDividend)),col($(inputColDivisor))).as($(outputCol)))
  }

  override def transformSchema(schema: StructType): StructType = {

    val dividendDataType = schema($(inputColDividend)).dataType
    require(dividendDataType.equals(DataTypes.DoubleType),
      s"Column ${$(inputColDividend)} must be DoubleType but was actually $dividendDataType.")

    val divisorDataType = schema($(inputColDivisor)).dataType
    require(divisorDataType.equals(DataTypes.DoubleType),
      s"Column ${$(inputColDivisor)} must be DoubleType but was actually $divisorDataType.")

    schema.add(StructField($(outputCol), DataTypes.DoubleType, false))

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}