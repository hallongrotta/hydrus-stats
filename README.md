# hydrus-stats
CLI tool to calculate various stats from Hydrus Network

Does two things at the moment, calculate the [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information) for pairs of tags, and calculates the [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) score for images based on their tags.

## Usage

Install with `pip` (I recommend using a venv to avoid package conflicts)

`pip install .`

The CLI can be run like this:

`hydrus-stats mi tentacles`

`hydrus-stats tfidf tentacles`

Multiple tags can also be entered, which will search for images that contain all tags.

`hydrus-stats tfidf tentacles breasts`

If you want to speed things up, the CLI can be invoked with args to use previously saved data files, like so

`hydrus-stats mi --cooccurrences stats/cooccurrences.pickle`

All files will be stored in `.stats/`

## What do these numbers/outputs mean?

The Partial Mutual Information (PMI), somewhat simply put, is a measure of how likely it is to observe another tag if an image already has a given tag.
This is not the same as a joint probability, however, since it factors in that some tags may be very common, and it would not be very informative to observe
them together with anything. The score is normalized to lie in the range of -1 to 1. 1 means that the tags always cooccur, and might be considered as siblings or as a parent/child pair. -1 means that the tags occur more often in general than they do together. 

The TF-IDF score is based on similiar ideas of information theory. TF-IDF is a weighted score indicating how 'important' words are, given how often they appear.
In this particular implementation, what is produced is a list of image hashes sorted, in ascending order, by the sum 
TF-IDF of their tags. The sum score can be seen as a measure of how much information is stored in the tag set. If an image has a very low score, then it is
very likely that the image is tagged with a single common tag, or a set of very common tags.

## A warning

Calculating cooccurrences can take a long time, especially for searches with many tags. Consider increasing the `min_count` parameter in `main.py` to decrease the number of "noise" tags, or make a more narrow search.
