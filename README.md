# Spotify-Project
Group project for AC 209 Data Science I

Tia Scarpelli, Daniel Varon, Tim Clements 

# Problem Statement and Motivation
Our goal is to predict the number of followers a Spotify playlist will garner based on song-level categorical features and quantitative audio features. Once we have a working predictive model, we plan to use it to build successful playlists based on a genre search filter. We will write an algorithm to assemble a large number of playlists from songs of a user-specified genre, and then use our model to identify the best playlist by predicting the success of each one.

# Introduction and Description of Data

We assembled a dataset of 1628 playlists totaling 85,313 songs using the python [Spotify API](https://github.com/plamere/spotipy). We downloaded playlists created by Spotify, as these are the most visible playlists on the platform. Our dataset contains qualitative features, such as key, time signature, mode, if the track has explicit language, etc., and quantitative audio features for each song, like acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, and valence.

Starting with the song-level data, we aggregated to the playlist level by performing mean aggregation on the quantitative features and majority-vote aggregation on the categorical features. We are now working with a data frame consisting of 1628 rows for 1628 playlists and ∼20 column features. These include the features listed above, as well as a quantitive feature describing the number of tracks on each playlist, and six categorical features describing the use of certain key words in track titles in the playlist—e.g., “remix,” “deluxe,” etc.

The Spotify API does not provide song-level genre information, but it does provide genre information at the level of the artist. We queried the genres of each artist. Our dataset has 1,230 unique genres. We mapped artist genres to song genres, and from there to playlist genres. 

# Literature Review/Related Work

# Modeling Approachand Project Trajectory
After constructing the playlist data frame, we explored the influence of different predictors on the number of followers (the response variable). The first thing we noticed is that the response variable is heavily imbalanced (right-skewed), with a small number of extremely popular playlists (Fig. 1). We therefore plan to predict the log(followers) for a given playlist, rather than the absolute number of followers.

# Results, Conclusions, and Future Work

Future work: 
* Download more playlists from users other than Spotify. 
* Inlcude playlists that were created by people (from websites such as “The Art of the Mix”)
* Get user artists lists because artists co-occurring in a user's collection are probably linked. 
* Get associated acts from Wikipedia to build playlists with similar artists 
