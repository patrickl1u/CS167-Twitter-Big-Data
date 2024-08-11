#!/bin/bash

# assume environment is set up properly

mvn clean
mvn package

#spark-submit --class edu.ucr.cs.cs167.twitterproject.App --master "local[*]" target/twitterproject-1.0-SNAPSHOT.jar data/Tweets_1k.json.bz2
# data is located in `data/` subdirectory, may need to change
mvn scala:run -DmainClass=edu.ucr.cs.cs167.twitterproject.App -DaddArgs=data/Tweets_1k.json.bz2
