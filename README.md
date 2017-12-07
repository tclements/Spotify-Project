# Spotify-Project
Group project for AC 209 Data Science I

Tia Scarpelli, Daniel Varon, Tim Clements 

# Problem Statement and Motivation
Our goal is to predict the number of followers a Spotify playlist will garner based on song-level categorical features and quantitative audio features. Once we have a working predictive model, we plan to use it to build successful playlists based on a genre search filter. We will write an algorithm to assemble a large number of playlists from songs of a user-specified genre, and then use our model to identify the best playlist by predicting the success of each one.

# Introduction and Description of Data

We assembled a dataset of 1628 playlists totaling 85,313 songs using the python [Spotify API](https://github.com/plamere/spotipy). We downloaded playlists created by Spotify, as these are the most visible playlists on the platform. Our dataset contains qualitative features, such as key, time signature, mode, if the track has explicit language, etc., and quantitative audio features for each song, like acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, and valence.

Starting with the song-level data, we aggregated to the playlist level by performing mean aggregation on the quantitative features and majority-vote aggregation on the categorical features. We are now working with a data frame consisting of 1628 rows for 1628 playlists and ∼20 column features. These include the features listed above, as well as a quantitive feature describing the number of tracks on each playlist, and six categorical features describing the use of certain key words in track titles in the playlist—e.g., “remix,” “deluxe,” etc.

The Spotify API does not provide song-level genre information, but it does provide genre information at the level of the artist. We queried the genres of each artist. We mapped artist genres to song genres based on first artist listed for each song, and from there to playlist genres considering the genre of each song in the playlist. Our dataset has 1,230 unique genres and many artists have more than one genre listed. The most popular and recognizable genres, 26 in total, were determined based on occurence in the entire dataset, and indicator variables were made for each of these popular genres. A positive observation was given for each genre that was listed for an artist (so many artists/songs had multiple positive observations). These indicator variables were then treated the same as all other categorical variables by using majority-vote aggregation by playlist; for example, if a majority of the tracks in a playlist have rock listed as a genre then that playlist was given a positive observation for rock.  

# Literature Review/Related Work
The success of our models to predict playlist popularity is dependent on being able to discriminate between the audio features of each playlist. The Echo Nest, a company that Spotify aquired in 2014, has done much of the work on algorithmically determining the audio features of a song. The [Echo Nest's Analyzer API](http://docs.echonest.com.s3-website-us-east-1.amazonaws.com/_static/AnalyzeDocumentation.pdf), which is now part of the Spotipy API, could identify a song's musical content such as tempo, loudness, timbre and sections, which are defined as  "large variations in rhythm or timbre, e.g. chorus, verse, bridge, guitar solo, etc..".  These quantities can be combined to form a song's second-order attributes such as energy or danceability. The [Spotify API audio features](https://developer.spotify.com/web-api/get-audio-features/) method, which supplies second-order song attributes for each song, has been very helpful in improving our playlist-popularity regression.

Creating new playlists requires finding songs that are similar. The earliest work on machine playlist creation was in quantitatively fnding similarity between songs. Logan and Salomon, 2001 implemented an alogrithm to find the "distance" between songs by computing the spectral features of each song. Songs with similar spectral features, or signatures, were most likely similar. This method was able to find songs that were either in the same genre, by the same artist or on the same album. We use a similar distance measure create playlists for each genre. Berenzweig et al., 2003 extended the work of Logan and Salomon, by including subjective information about artist similarity from surveys of humans, and web scraping music data bases and online playlists. They used these data to build similarity matrices that give the probability of two artists being on the same playlists. We use both strategies, computing the distance between songs and building a similarity matrix to build our playlists. 

# Modeling Approach and Project Trajectory
After constructing the playlist data frame, we explored the influence of different predictors on the number of followers (the response variable). The first thing we noticed is that the response variable is heavily imbalanced (right-skewed), with a small number of extremely popular playlists:

![logfollowers](/FIGURES/logfollowers.png)

 We therefore predict the log(followers) for a given playlist, rather than the absolute number of followers. Playlists with zero followers (26) were removed from the dataset because it was assumed that these playlists may be not representative of the population of followed playlists - they may be intended to be listened to without users actually choosing to follow them (i.e. soothing background noise).

Interaction terms were included based on observation of predictor interaction. Loudness influenced the spread of other song characteristics and danceability was influenced by the speechiness and tempo of the song (see ipython notebook for more details).

## KNN
We started with KNN, as it is one of the simplest regression models. As a baseline, a KNN model was fit to all predictors in our data set to predict the log of followers for each playlist. This model performed poorly, with an R<sup>2</sup> of 0.05 on the testing set. We then fit a KNN model on the quantitative features for each playlist. This increased the R<sup>2</sup> to 0.23. We finally included the interaction terms of loudness and danceablilty for the final KNN regression, as we identified these as important predictors in our EDA. This increased the R<sup>2</sup> once again to 0.33.

![KNN Regression](/FIGURES/KNN_REGRESSION.png)

## Regularized (Ridge) Regression 
The next approach we evaluated was Ridge regression. We first fit the log(followers) using only playlist popularity as predictor. This yielded an R<sup>2</sup> score of 0.12 on the test set. We then added in all of the predictors, as well as second-order polynomial features for the continuous predictors, and interaction terms between loudness, danceability, speechiness, and tempo, as our EDA revealed these to have important interactions. The test R<sup>2</sup> score in this case soared to 0.48. We used cross-validation to estimate the optimal regularization parameter, which we found to be 0.1. The mean cross-validation score on the training set for this model was 0.45.

We determined which predictors are most important by computing the absolute magnitudes of their fitted coefficients. These are the top 30 predictors:
![ridge_top](/FIGURES/ridge_top_predictors.png)

These are the bottom 30 predictors:
![ridge_bot](/FIGURES/ridge_bot_predictors.png)

## Gradient Boosted Regression Tree
Gradient Boosted Regression was used to predict playlists to see if we could improve on typical methods of regression. All quantitative and indicator variables were used as predictors, including interaction terms discussed above. Data was split into test and train sets, and the training set was used to fit the model. Cross validation was used to determine the optimal model fit which required a grid search for learning rate (range 0.1-0.01), maximum depth (4-12), minimum samples per leaf (3-7) and maximum number of features used for split decision (0.3-0.6). The optimal parameters were a learning rate of 0.01, maximum depth of 12, maximum number of features used of 0.3 and minimum samples per leaf of 7. The high value for minimum samples per leaf may in part have been chosen to compensate for the high maximum depth. Then a second grid search with cross validation was performed to find the optimal learning rate given the optimal parameters (range of 0.1-0.001 was used). The optimal learning rate remained 0.01, resulting in an $R^2$ score of 0.97 which is very high, likely due to overfitting to the training set which is shown by the wide separation between test and train error in the deviance plot shown below.  

![Deviance Plot](/FIGURES/GBRTree_deviance_plot)

The most important quantitative predictors for tree splits were popularity and number of tracks, shown in the figure below, as expected because users will generally want popular songs and a large number of songs in a playlist. The interaction of these predictors with loudness were also important. The most important indicator variables (which will generally always have a lower number of splits because there are only 2 options to choose to split on 1 or 0) were genre related - pop and rock - and word related - mention of remastered or remix in song titles.

![Predictor Importance](/FIGURES/GBRTree_predictors)

## MLP
The new method that was implemented was a neural networks method called Multi-layer Perceptron and there is a sklearn package called MLPRegressor that can be used to implement MLP regression. The process was very similar to gradient boosted regression in that parameters were fit in a two step process. The regularization parameter (alpha, range 0.1-0.00001) was first fit and then activation which is the function used for the hidden layer and the number of hidden layers (range 50-200). The optimal parameters were a regularization parameter of 0.1, a logistic function for the activation function and using 200 hidden layers. 

# Results, Conclusions, and Future Work

|                | Cross validation score | Training set R<sup>2</sup> | Test set R<sup>2</sup> |
|----------------|------------------------|-----------------|-------------|
| KNN            |      0.35              |                 |             |
| Ridge          |      0.45              |    0.56         |  0.48       |
| Gradient Boost |      0.55              |                 |             |
| MLP            |      0.44              |                 |             |

Future work: 
* Download more playlists from users other than Spotify. 
* Inlcude playlists that were created by people (from websites such as “The Art of the Mix”)
* Get user artists lists because artists co-occurring in a user's collection are probably linked. 
* Get associated acts from Wikipedia to build playlists with similar artists 
* Look for more predictors to capture playlists that have very low numbers of followers (possibly: how playlists are titled, how playlists are made available to users, etc.)
