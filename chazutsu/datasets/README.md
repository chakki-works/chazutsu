# DataSets

Let me introduce supported dataset in chazutsu.  

## [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

This dataset offers 3 type of labeled dataset.

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

**Dataset File format**

* label (polarity, rating, subjectivity)
* text

**Papers (from official site)**

* Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan, [Thumbs up? Sentiment Classification using Machine Learning Techniques](http://www.cs.cornell.edu/home/llee/papers/sentiment.home.html), Proceedings of EMNLP 2002.
* Bo Pang and Lillian Lee, [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](A sentimental education: Sentiment analysis using subjectivity), Proceedings of ACL 2004.
* Bo Pang and Lillian Lee, [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.home.html), Proceedings of ACL 2005.

## [Customer Review Datasets](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets)

This dataset offers annotated product's review. The annotation is done as below:

* [t]: title
* xxxx[+n|-n]: xxxx is feature, and evaluation (+|-) is expressed by 1-3
* [p]: describe about feature but you have to interpret pronoun
* [u]: does not describe about feature
* [s]: suggestion or recommendation
* [cc]: comparison with a competing product from a different brand
* [cs]: comparison with a competing product from the same brand

If you want to download this dataset, please direct as following.

* `CustomerReview.5products`: download the dataset that annotated to 5 products
* `CustomerReview.9additional`: download the dataset that annotated to 9 additional products
* `CustomerReview.3more`: download the dataset that annotated to 3 more products

(If you want to use `9additional` or `3more`, [you have to intall `unrar` or `bsdtar` to unpack rarfile](https://github.com/markokr/rarfile/blob/master/README.rst))

**Dataset File format**

* sentence-type: t, po (has polarity), -(the sentence that has no annotation)
* polarity: -3 ~ 3. "po" sentence-type has this value
  * if sentence includes multiple score(feature), take average
* detail: (feature)_(polarity)_(attribute) ex: size_-2_u, digital zoom_+1_
  * if attributes (p, u, s, cc, cs) do not appear, attribute is blank
* sentence

**Papers (from official site)**

* Hu and Liu, KDD-2004, [Mining and Summarizing Customer Reviews](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf)
* Ding, Liu and Yu, WSDM-2008, [A Holistic Lexicon-Based Approach to Opinion Mining](https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf)
* Liu et al., IJCAI-2015, [Automated Rule Selection for Aspect Extraction in Opinion Mining](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/10766/10842)


## [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)

This dataset offers 20 newsgroups collection. Some of the categories are related, so these partitioned by 6 group.

* comp: comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x
* rec: rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
* sci: sci.crypt, sci.electronics, sci.med, sci.space
* misc: misc.forsale
* politics: talk.politics.misc, talk.politics.guns, talk.politics.mideast
* religion: talk.religion.misc, alt.atheism, soc.religion.christian

For example, `alt.atheism` and `talk.religion.misc`, `comp.windows.x` and `comp.graphics` are close, `rec.sport.baseball` and `sci.crypt` is far.

If you want to download this dataset, please user below class.

* `NewsGroup20`: the version that excludes duplicates and includes only "From" and "Subject" headers.

**Dataset File format**

* news group name
* category name
* subject(Subject)
* author(From)
* text

**Papers (from official site)**

* [Newsweeder: Learning to filter netnews](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.22.6286)
