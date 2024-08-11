package edu.ucr.cs.cs167.twitterproject

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.SaveMode
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, concat_ws, lit}

object App {
  def main(args: Array[String]) {
    val conf = new SparkConf
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")
    val spark = SparkSession
      .builder()
      .appName("CS167_Twitter_Analysis_App")
      .config(conf)
      .getOrCreate()
    try {
      import spark.implicits._
      val t1 = System.nanoTime
      val inputFile: String = args(0)

      // ########
      // part 1
      // ########

      var validOperation = true

      val tweetsDF = spark.read.format("json") // loads json input file as our dataframe
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)

      import spark.implicits._
      tweetsDF.createOrReplaceTempView("root") // create view for dataframe

      // sql query that filters attributes
      // also renames entities.hashtags.text as hashtags and user.description as user_description to match sample output
      val query: String = "SELECT id, text, entities.hashtags.text as hashtags, user.description as user_description, retweet_count, reply_count, quoted_status_id FROM root;"
      val schema = spark.sql(query)
      schema.write.mode(SaveMode.Overwrite).json("tweets_clean") // outputs results to tweets_clean output json file
      schema.printSchema() // prints dataframe schema

      schema.createOrReplaceTempView("tweets") // create new view for the dataframe

      // top-k slq query that counts top 20 hashtags
      val SQLQuery =
        """
                  SELECT hashtag, COUNT(*) AS count
                  FROM (
                    SELECT EXPLODE(hashtags) AS hashtag
                    FROM tweets
                  )
                  GROUP BY hashtag
                  ORDER BY count DESC
                  LIMIT 20
                  """
      val array = spark.sql(SQLQuery) // create an array to collect the top 20 words
        .collect()
        .map(row => row.getString(0))

      println(array.mkString(", ")) // prints the array

      // ########
      // end part 1, begin part 2
      // ########

      // compute intersection between list of hashtags and list of topics
      // keep only records that have a topic
      val tweets_top20topics = schema.filter(size(array_intersect(col("hashtags"),lit(array))) > 0)

      // replace 'hashtags' attribute with 'topic' attribute
      // keep only first element of result from 'array_intersect'
      tweets_top20topics.createOrReplaceTempView("tweets")
      val tweets_topicDF = tweets_top20topics.withColumn("topic", col("hashtags")(0)).drop(tweets_top20topics("hashtags"))

      // save output as JSON in file called "tweets_topic"
      tweets_topicDF.write.mode(SaveMode.Overwrite).json("tweets_topic")
      tweets_topicDF.show()

      // ########
      // end part 2, begin part 3
      // ########

      var twitterData: DataFrame = tweets_topicDF

      //creating new column that combines text and user_description
      twitterData = twitterData.withColumn("text_desc", concat_ws(",", col("text"), col("user_description")))

      //tokenizing text_desc column into tokens/words
      val tokenizer = new Tokenizer().setInputCol("text_desc").setOutputCol("words")

      //converts the tokens/words into a set of numeric features
      val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features")

      //using hashtag because for some reason the sample file has hashtag instead of topic
      //converts each unique hashtag/topic into a label --> multiple labels = multiple classes
      //maps each hashtag/topic to integer values
      val stringIndexer = new StringIndexer().setInputCol("topic").setOutputCol("label").setHandleInvalid("skip")

      //predicts the hashtag/topic from set of features
      //default configuration uses features and labels
      //default number of iterations is 100
      val logisticRegression = new LogisticRegression().setMaxIter(100)

      val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, stringIndexer, logisticRegression))

      val paramGrid: Array[ParamMap] = new ParamGridBuilder() //tells validator the different parameters to try
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000)) //10,100, 1000, try these three and choose the best
        .addGrid(logisticRegression.regParam, Array(0.01, 0.1, 0.3, 0.8)) //0.01, 0.1, 0.3, 0.8, try these 4, choose the best, maybe try small and big values
        .build()

      val cv = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label")) //try combinations
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(0.8)
        .setParallelism(2)

      val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = twitterData.randomSplit(Array(0.8, 0.2))

      // Run cross-validation, and choose the best set of parameters.
      val logisticModel: TrainValidationSplitModel = cv.fit(trainingData)

      val numFeatures: Int = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[HashingTF].getNumFeatures
      val regParam: Double = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getRegParam
      println(s"Number of features in the best model = $numFeatures")
      println(s"RegParam the best model = $regParam")

      val predictions: DataFrame = logisticModel.transform(testData)
      predictions.select("id", "text", "topic", "user_description", "label", "prediction").show()

      val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")

      val accuracy: Double = evaluator.evaluate(predictions)
      println(s"Accuracy of the test set is $accuracy")

      val metrics = evaluator.getMetrics(predictions)
      println(s"Precision: ${metrics.weightedPrecision}")
      println(s"Recall: ${metrics.weightedRecall}")

      // ########
      // end part 3
      // ########

      val t2 = System.nanoTime
      println(s"Execution finished in ${(t2 - t1) * 1E-9} seconds")

    } finally {
      spark.stop
    }
  }
}
