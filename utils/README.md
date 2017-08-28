# Parsing Model

## Parser
The parsing model is based on the Dozat and Manning graph-based parser.
The inputs to the parser are:

* Word Identities
* Predicted POS Identities
* Predicted Stag Identities

We will try and add more features later. In particular, the Stanford parsing system for UD adds morphological information using the char-LSTM with attention.

All the inputs listed above are sequences of length equal to the sentence length; our placeholders in the TensorFlow implementation are just those lists! Simple.

## Notes on the Implementation

### One-sentence-per-line Format
The input format is the one-sentence-per-line format. However, we also add the '<-root->' token at the beginning of each line. We might change the code so that we add '<-root->' in a later process, but this way avoids a potential bug. '<-root->' should be distinguished from the arc label 'root'.

### Data Loader
``data_process_secsplit.py`` serves as the universal data loader.
test_opts is ``None`` when you train the model. 

We do not reserve the zero index for the arc labels (rels); zero is used both for zero-padding and the most frequent label. This is because we can extract where those paddings are by looking at zero's in word sequences, and we do not

