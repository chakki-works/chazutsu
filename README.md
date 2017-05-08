# chazutsu

Do you have trouble with gathering the data for natural language processing?  
For example, exploring the kinds of data, finding where to download, handling huge size and parsing its data (and more!).

**chazutsu** helps you to fighting above problems.

<img src="https://github.com/chakki-works/chazutsu/raw/master/docs/chazutsu.png" width="50">

*[photo from Kaikado, traditional Japanese chazutsu maker](http://www.kaikado.jp/english/goods/design.html)*

## Install

```
pip install chazutsu
```

## How to use

**chazutsu** supports you from data download to making file that can be read by [pandas](http://pandas.pydata.org/) etc.

![feature.png](./docs/feature.png)

### Download the Datasets

You can download the datasets by chazutsu like following.

```py
>>>from chazutsu.datasets import datasets
>>>datasets.list("language_model")
Penn Treebank(PTB): POS Annotated data (size: xx kb)
...
>>>datasets.download.PTB(directory="my/dataset/ptb/")
ptb.txt is saved to my/dataset/ptb/
```

The list of datasets are described in `datasets/README.md`. You can confirm it on GitHub too (and if you send the pull request, dataset list is updated!).

### Save to BigQuery

You can download the datasets as table of [BigQuery](https://cloud.google.com/bigquery/) by chazutsu like following.

```py
>>>from chazutsu.datasets import datasets
>>>datasets.auth("project_id", "account", "paht/to/key.pem")
Connect to BigQuery
>>>datasets.download.PTB(directory="bq:table_name")
PTB data is inserted into table_name
```

### Split to files

If the file size is too large to deal with, you can split to some files.

```py
>>>from chazutsu.datasets import datasets
>>>datasets.download.PTB(directory="my/dataset/ptb/", each_size="200")
ptb1.txt, ptb2.txt, ptb3.txt is saved to my/dataset/ptb/ (each size is 200kb)
```

You can use "M" and "G" to direct the file size.

### Make sample file

You don't want to load the all dataset to watch the some lines of data!

If you direct the `sample_size`, you can get the file that is sampled from dataset.

```py
>>>from chazutsu.datasets import datasets
>>>datasets.download.PTB(directory="my/dataset/ptb/", sample_size=500)
ptb.txt is saved to my/dataset/ptb/
ptb_sample.txt is also saved (includes 5oo samples)
```

