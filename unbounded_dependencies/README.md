## Unbounded Dependency Data Evaluation
---

### Data
- Unbounded Dependency Data from [Rimell et al. 2009](http://www.aclweb.org/anthology/D/D09/D09-1085.pdf)
<!--- - We found one sentence in the raw data ``longrange-distrib/sbj_extract_rel_clause/dev.raw.sbj_extract_rel_clause`` not annotated with an unbounded dependency: 

Then , my mother blushed at this small lie ; ; for she knew and we knew that it was cowardice that had made one more radish that night just too impossible a strain .

We deleted that sentence for evaluation. There is also disagreement in punctuation between the raw files and the annotated files (-LRB- etc.). Therefore, we created a new sentence file for each construction type.--!>

### TAG Parser Evaluation
<!--- - In the TAG representations, all the other constructions but RNR can be resolved by the relative clause operation proposed in [Xu et al. 2017](http://www.aclweb.org/anthology/W/W17/W17-6214.pdf).---!>
**Transformation procedures**
1. Relative Clause. See [Xu et al. 2017](http://www.aclweb.org/anthology/W/W17/W17-6214.pdf).
2. Coordination. See [Xu et al. 2017](http://www.aclweb.org/anthology/W/W17/W17-6214.pdf).
3. Prepositional objects have to been treated in a special way.
    * If a word is assigned tCO (co-anchor), add the same edges as the head
    * If a word is a preposition that modifies
4. Copula. First stem the sentence using nltk.
**Label conversion**
* nsubj -> 0
* dobj -> 1
* pobj -> 1 (Need to check)
* nsubjpass -> 1
* advmod -> -unk-
* prep -> ADJ

### Stanford Dependency Evaluation
- Direct
- Indirect
- Complex

