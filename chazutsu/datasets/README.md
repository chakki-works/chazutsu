# DataSets

Let me introduce supported dataset in chazutsu.  

## [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

It offers 3 type of labeled dataset.

* positive/negative (1000pos/1000neg)
* subjective rating (like "two and a half stars")
  * rating in the dataset is normalized. 0-to-4 star is converted to 0.1 - 0.9, and 0-to-5 star is converted to 0-to-1.
  * see also "Label Decision" in [readme](http://www.cs.cornell.edu/people/pabo/movie-review-data/scaledata.README.1.0.txt)
* subjective or objective (5000sub/5000obj)

If you want to download this dataset, please direct the kind as following.

* `MovieReview.polarity`
 * `MovieReview.polarity_v1`: includes v1 & v2 data(5331 positive and 5331 negative)
* `MovieReview.rating`
* `MovieReview.subjectivity`

