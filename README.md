# Predicting precise location of 2'O methylations with Machine Learning and Deep Learning models
[Link to the first paper](https://ieeexplore.ieee.org/abstract/document/8512780)

### This repository contains codes for two papers. 
### The second paper is under revision and will be added as soon as it gets accepted.

#### Requirments:
[Link to dna2vec](https://github.com/pnpnpn/dna2vec)\
[deepexlain for extracting significant scores around 2'O sites](https://github.com/marcoancona/DeepExplain)
### The impact of three key factors are investigated:
* Length of RNA sequences containing 2'O sites
* Embedding method (one-hot encoding vs dna2vec)
* Type of predictive models (SVM and CNN are chosen as the best Machine Learning and Deep Learning models, respectively)

## Main results
### Embedding has no impact on CNN performance however it makes a considerable difference in SVM models performance
### Increasing input sequence length does not have any impact.
### Significant nucleotides around 2'O sites are extracted by attention mechanism of CNN
* The location of 2'O motif reported in the [nm_seq](https://www.ncbi.nlm.nih.gov/pubmed/28504680) has got highest attention from CNN
