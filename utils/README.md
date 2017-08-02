## Parsing Model

The parsing model is based on the Dozat and Manning graph-based parser.
The inputs to the parser are:

* Word Identities
* Predicted POS Identities
* Predicted Stag Identities

We will try and add more features later. In particular, the Stanford parsing system for UD adds morphological information using the char-LSTM with attention.

All the inputs listed above are sequences of length equal to the sentence length; our placeholders in the TensorFlow implementation are just those lists! Simple.

