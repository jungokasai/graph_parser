# A TensorFlow implementation of Graph-based Biaffine Parser


### Table of Contents  
* [Requirements](#requirements)  
* [GloVe](#glove)
* [Train a Parser](#train)
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
The supertagger takes as input a file in the Conllu+Supertag (conllustag) format, in which one column for supertags is added to the original conllu format at the end. See a [sample](sample_data/conllu/sample.conllustag).

## <a name="train"></a>Train a Parser
All you need to do is to create a new directory for your data in the [conllustag format](#Data Format) and a json file for the model configuration and data information. We provide a [sample json file](sample_data/config_demo.json) for the [sample](sample_data) data directory. You can train a supertagger on the sample data by the following command:
```bash
python train_graph_parser.py sample_data/config_demo.json
```
After running this command, you should be getting the following files and directories in sample_data/:


## <a name="jackknife"></a>Jackknife POS tagging
To Be Added.


## <a name="pretrained"></a>Run a pre-trained TAG Supertagger

To Be Added.

<!-- ## Notes

If you use this tool for your research, please consider citing:
```
@InProceedings{Kasai&al.17,
  author =  {Jungo Kasai and Robert Frank and R. Thomas McCoy and Owen Rambow and Alexis Nasr},
  title =   {TAG Parsing with Neural Networks and Vector Representations of Supertags},
  year =    {2017},  
  booktitle =   {Proceedings of EMNLP},  
  publisher =   {Association for Computational Linguistics},
}
```
-->
