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
* review

**Citation/License**

* `polarity`: Bo Pang and Lillian Lee, [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://www.cs.cornell.edu/home/llee/papers/cutsent.home.html), Proceedings of ACL 2004.
 * `polarity_v1`: * Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan, [Thumbs up? Sentiment Classification using Machine Learning Techniques](http://www.cs.cornell.edu/home/llee/papers/sentiment.home.html), Proceedings of EMNLP 2002.
* `rating`: Bo Pang and Lillian Lee, [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.home.html), Proceedings of ACL 2005.
* `subjectivity`: [A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts](http://www.cs.cornell.edu/home/llee/papers/cutsent.home.html), Proceedings of ACL 2004.

## [Customer Review Datasets](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets)

This dataset offers annotated product's review. The annotation is done as below:

* [t] : title
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
* review

**Citation/License**

* `CustomerReview.5products`: Minqing Hu and Bing Liu. [Mining and Summarizing Customer Reviews](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf), Proceedings of KDD-2004
* `CustomerReview.9additional`: Xiaowen Ding, Bing Liu and Philip S. Yu. [A Holistic Lexicon-Based Approach to Opinion Mining](https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf), Proceedings of WSDM-2008
* `CustomerReview.3more`: Qian Liu, Zhiqiang Gao, Bing Liu and Yuanlin Zhang. [Automated Rule Selection for Aspect Extraction in Opinion Mining](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/10766/10842), Proceedings of IJCAI-2015

## [Large Movie Review Dataset(IMDB)](http://ai.stanford.edu/~amaas/data/sentiment/)

This dataset offers 25,000 train/test movie reviews that have positive/negative annotation.

If you want to download this dataset, please use below class.

* `IMDB`

This dataset contains additional unlabeled data for unsupervised learning. You can access this data as following.

```python
r = chazutsu.datasets.IMDB().download()
r.unlabeled_data()
```

**Dataset File format**

* label: positive(=1)/negative(=0)
* rating: 1~10 (from file name)
* review

(unsupervised only have review)

**Citation/License**

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf), ACL 2011

## [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)

This dataset offers 20 newsgroups collection. Some of the categories are related, so these partitioned by 6 group.

* comp: comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x
* rec: rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
* sci: sci.crypt, sci.electronics, sci.med, sci.space
* misc: misc.forsale
* politics: talk.politics.misc, talk.politics.guns, talk.politics.mideast
* religion: talk.religion.misc, alt.atheism, soc.religion.christian

For example, `alt.atheism` and `talk.religion.misc`, `comp.windows.x` and `comp.graphics` are close, `rec.sport.baseball` and `sci.crypt` is far.

If you want to download this dataset, please use below class.

* `NewsGroup20`: the version that excludes duplicates and includes only "From" and "Subject" headers.

**Dataset File format**

* group
* group category
* subject(Subject)
* author(From)
* text

**Citation/License?**

* [Newsweeder: Learning to filter netnews](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.22.6286)

(could not find official cite & license)

## [Reuters News Courpus (RCV1-v2)](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)

This dataset offers Reuters News articles that are categorized by topic, industries and region. It has 23,149 train and 781,265 test documents.

 If you want to download this dataset, please use below class.

* `ReutersNews.topics`: download the topic annotated data
* `ReutersNews.industries`: download the industries annotated data
* `ReutersNews.region`: download the region annotated data

*(You have to take care that the each document has multiple labels)*

**Dataset File format**

`topics`

* topic category
* parent topic category
* tokenized document

`industries`

* industries category
* 2nd-level category (10 category expressed by 2 character)
* tokenized document

`region`

* region
* tokenized document

**Citation/License**

* Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. [RCV1: A New Benchmark Collection for Text Categorization Research](http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). Journal of Machine Learning Research, 5:361-397, 2004.

Because of the agreement of RCV1 CD-ROMs from Reuters, the original data should not  be reconstructed. So the dataset removes the large stop words and replace the remaining words with stems, and scramble the order of the stems.

Please refer the detail on the [official site](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm).


## [Penn Tree Bank](https://github.com/tomsercu/lstm)

The original [Penn Tree Bank](http://aclweb.org/anthology/J93-2004) has linguistic structure annotations (and it is not free).  
So here offers the variant dataset from [tomsercu/lstm](https://github.com/tomsercu/lstm),
that omits the annotation and splited to train/valid/test dataset.

If you want to download this dataset, please use below class.

* `PTB`

**Dataset File format**

It downloads raw text file. You can tokenize it from returned object.

```python
>>>import chazutsu
>>>r = chazutsu.datasets.PTB().download()
>>>tokenized, vocab = r.tokenize("valid")  # train, test, or valid
>>>print(tokenized[:10])
[647, 135, 320, 5, 468, 58, 2561, 6, 256, 2530]
>>>rev_vocab = {v:k for k, v in vocab.items()}
>>>print([rev_vocab[i] for i in tokenized[:10]])
['consumers', 'may', 'want', 'to', 'move', 'their', 'telephones', 'a', 'little', 'closer']
```

**Citation/License**

* Apache License 2.0
* [RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/pdf/1409.2329v4.pdf)

## [WikiText-2](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)

WikiText-2 is the language modeling dataset that is larger than PTB (over 2 times!).  
This dataset is created from Wikipedia articles that are verified Good and Featured.

If you want to download this dataset, please use below class.

* `WikiText2`

**Dataset File format**

It downloads raw text file. You can tokenize it from returned object (same as PTB).

**Citation/License**

* Creative Commons Attribution-ShareAlike License
* Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)


## [WikiText-103](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)

WikiText-2 is the language modeling dataset that is larger than PTB (over 110 times!).  
This dataset is created from Wikipedia articles that are verified Good and Featured.

If you want to download this dataset, please use below class.

* `WikiText103`

**Dataset File format**

It downloads raw text file. You can tokenize it from returned object (same as PTB).

**Citation/License**

* Creative Commons Attribution-ShareAlike License
* Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)

## [text8](http://mattmahoney.net/dc/textdata)

This dataset offers cleaned enwik9. The enwik9 is the first 10^9 bytes of the English Wikipedia dump on Mar. 3, 2006.  
The text8 is compressed version of enwik9 by discarding the outside of the <text> tag, removing some tags and so on (you can confirm these process on the official site (enwik9 -> 17compressors -> fil9 -> extra 8 compressors -> text8!).

If you want to download this dataset, please use below class.

* `Text8`

In this dataset, `test_size` is treated as bytes (mega byte) for splitting the file. Default is 10Mb accordings to [Learning Longer Memory in Recurrent Neural Networks](https://arxiv.org/abs/1412.7753).

**Dataset File format**

It downloads raw text file. You can tokenize it from returned object.

```
r = chazutsu.datasets.Text8().download(directory=DATA_ROOT)
tokenized, vocab = r.tokenize("test", min_word_count=5)
```

I recommend to set `min_word_count` 5, same as [Learning Longer Memory in Recurrent Neural Networks](https://arxiv.org/abs/1412.7753).

**Citation/License?**

* [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html)
* [Learning Longer Memory in Recurrent Neural Networks](https://arxiv.org/abs/1412.7753)
  * If you use this dataset, it'll be good to preprocess the data according to this paper.
