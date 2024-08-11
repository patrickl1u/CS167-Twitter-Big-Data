package edu.ucr.cs.cs167.barci003

import org.apache.spark.sql.{SaveMode, SparkSession, DataFrame, Row}
import org.apache.spark.{SparkConf, sql}

object App {

  def main(args: Array[String]) {
    // Initialize Spark context

    val conf = new SparkConf()
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("CS167_Final_App")
      .config(conf)
      .getOrCreate()

    val inputFile: String = args(0)

    try {
      // Import Beast features
      val t1 = System.nanoTime()
      var validOperation = true

      val tweetsDF = spark.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)

      import spark.implicits._
      tweetsDF.createOrReplaceTempView("root")

      val query: String = "SELECT id, text, entities.hashtags.text as hashtags, user.description as user_description, retweet_count, reply_count, quoted_status_id FROM root;"
      val schema = spark.sql(query)
      schema.write.mode(SaveMode.Overwrite).json("tweets_clean")
      schema.printSchema()

      schema.createOrReplaceTempView("tweets")

      val SQLQuery = """
                  SELECT hashtag, COUNT(*) AS count
                  FROM (
                    SELECT EXPLODE(hashtags) AS hashtag
                    FROM tweets
                  )
                  GROUP BY hashtag
                  ORDER BY count DESC
                  LIMIT 20
                  """

      val array = spark.sql(SQLQuery)
        .collect()
        .map(row => row.getString(0))

      println(array.mkString(", "))

      val t2 = System.nanoTime()

      //println(s"Operation on file '$inputFile' took ${(t2 - t1) * 1E-9} seconds")
    } finally {
      spark.stop()
    }
  }
}
