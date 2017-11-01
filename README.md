# chazutsu

![chazutsu_top.PNG](./docs/chazutsu_top.PNG)  
*[photo from Kaikado, traditional Japanese chazutsu maker](http://www.kaikado.jp/english/goods/design.html)*

chazutsu is the dataset downloader for NLP.

```py
>>> import chazutsu
>>> r = chazutsu.datasets.IMDB().download()
>>> r.train_data().head(5)
```
Then

```
   polarity  rating                                             review
0         0       3  You'd think the first landing on the Moon woul...
1         1       9  I took a flyer in renting this movie but I got...
2         1      10  Sometimes I just want to laugh. Don't you? No ...
3         0       2  I knew it wasn't gunna work out between me and...
4         0       2  Sometimes I rest my head and think about the r...
```

You can use chazutsu on Jupyter.

## Install

```
pip install chazutsu
```

## Supported datasetd

chazutsu supports various kinds of datasets!  
**[Please see the details here!](https://github.com/chakki-works/chazutsu/tree/master/chazutsu)**

* Sentiment Analysis
  * Movie Review Data
  * Customer Review Datasets
  * Large Movie Review Dataset(IMDB)
* Text classification
  * 20 Newsgroups
  * Reuters News Courpus (RCV1-v2)
* Language Modeling
  * Penn Tree Bank
  * WikiText-2
  * WikiText-103
  * text8
* Text Summarization
  * DUC2003
  * DUC2004
  * Gigaword
* Textual entailment
  * The Multi-Genre Natural Language Inference (MultiNLI)


# How it works

chazutsu not only download the dataset, but execute expand archive file, shuffle, split, picking samples process also (You can disable the process by arguments if you don't need).

![chazutsu_process1.png](./docs/chazutsu_process1.png)

```
r = chazutsu.datasets.MovieReview.polarity(shuffle=False, test_size=0.3, sample_count=100).download()
```

* `shuffle`: The flag argument for executing shuffle or not(True/False).
* `test_size`: The ratio of the test dataset (If dataset already prepares train and test dataset, this value is ignored).
* `sample_count`: You can pick some samples from the dataset to avoid the editor freeze caused by the heavy text file.
* `force`: Don't use cache, re-download the dataset.

chazutsu supports fundamental process for tokenization.

![chazutsu_process2.png](./docs/chazutsu_process2.png)

```py
>>> import chazutsu
>>> r = chazutsu.datasets.MovieReview.subjectivity().download()
>>> r.train_data().head(3)
```

Then

```
    subjectivity                                             review
0             0  . . . works on some levels and is certainly wo...
1             1  the hulk is an anger fueled monster with incre...
2             1  when the skittish emma finds blood on her pill...
```

Now execute the tokenization.

```py
r_idx = r.to_indexed().make_vocab(min_word_count=3)
r_idx.train_data().head(3)
```

Then

```
   subjectivity                                             review
0             0  [240, 18, 73, 1824, 3, 7, 792, 306, 812, 30, 3...
1             1  [1, 6841, 7, 15, 3052, 4179, 1470, 12, 1910, 9...
2             1  [38, 1, 0, 2233, 142, 912, 18, 19, 0, 365, 116...
```

* `to_indexed`
  * `vocab_resources`: which resouce is used ("train", "valid", "test")
  * `vocab_columns`: whick column is used
* `make_vocab`
  * `tokenizer`: Tokenizer
  * `vocab_size`: Vocacbulary size
  * `min_word_count`: Minimum word count to include the vocabulary
  * `end_of_sentence`: If you want to clarify the end-of-line by specific tag, then use this.
  * `unknown`: The tag used for out of vocabulary word
  * `reserved_words`: The word that should included in vocabulary (ex. tag for padding)
  * `force`: Don't use cache, re-create the dataset.

## Additional Feature

### Use on Jupyter

You can use chazutsu on [Jupyter Notebook](http://jupyter.org/).  

![on_jupyter.png](./docs/on_jupyter.png)

Before you execute chazutsu on Jupyter, you have to enable widget extention by below command.

```
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```
