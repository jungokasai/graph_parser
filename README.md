# A TensorFlow implementation of Graph-based Biaffine Parser

<img src="/images/biaffine_parsing.png" width="300">

### Table of Contents  
* [Requirements](#requirements)  
* [GloVe](#glove)
* [Data Format](#data)
* [Train a Supertagger](#train)
* [Structure of the Code](#structure)
* [Jackknife POS Tagging](#jackknife)
* [Run a pre-trained TAG Supertagger](#pretrained)
* [Notes](#notes)

<!--* [Notes](#notes) -->

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 1.0.0 or higher is supported. 

## GloVe

Our architecture utilizes pre-trained word embedding vectors, [GloveVectors](http://nlp.stanford.edu/projects/glove/). Run the following:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip 
```
and save it to a sub-directory glovevector/. 

## Data Format
The biaffine parser takes as input a file in the Conllu+Supertag (conllustag) format, in which one column for supertags is added to the original conllu format at the end. See a [sample](sample_data/conllu/sample.conllustag).

## <a name="train"></a>Train a Parser
All you need to do is to create a new directory for your data in the conllustag format  and a json file for the model configuration and data information. We provide a [sample json file](sample_data/config_demo.json) for the [sample](sample_data) data directory. You can train a parser on the sample data by the following command:
```bash
python train_graph_parser.py sample_data/config_demo.json
```
After running this command, you should be getting the following files and directories in sample_data/:

| Directory/File | Description |
|------|--------|
|checkpoint.txt|Contains information about the best model.|
|sents/|Contains the words in the one-sentence-per-line format|
|gold_pos/|Contains the gold POS tags in the one-sentence-per-line format|
|gold_stag/|Contains the gold supertags in the one-sentence-per-line format|
|predicted_stag/|Contains the predicted supertags in the one-sentence-per-line format|
|Parsing_models/|Stores the best model.|
|conllu/sample.conllustag_stag|Contains the predicted supertags in the conllustag format|


## <a name="pretrained"></a>Run a pre-trained TAG Parser

To Be Added.

## Notes

If you use this tool for your research, please consider citing:
```
@InProceedings{Kasai&al.18,
  author =  {Jungo Kasai and Robert Frank and Pauli Xu and William Merrill and Owen Rambow},
  title =   {End-to-end Graph-based TAG Parsing with Neural Networks},
  year =    {2018},  
  booktitle =   {Proceedings of NAACL},  
  publisher =   {Association for Computational Linguistics},
}
```
