# Predictive algorithms from Electronic Health Records 

This repository hosts code for the working paper: *Exploring a complexity gradient in representation and predictive algorithms for EHRs* 

[**Documentation**](https://soda.gitlabpages.inria.fr/medical_embeddings_transfer)

[**Source Code**](https://gitlab.inria.fr/soda/medical_embeddings_transfer)

[**Working Paper repository**](https://github.com/strayMat/predictive_ehr_paper)

### Abstract

Electronic Health Records contain time-varying features with high cardinality.
Current state-of-the-art predictive models build on increasingly elaborated
pipelines --based on transformers-- to handle the complexity of these data.
Acknowledging the complexity to deploy, transfer and adapt these models on local
care environments, we explore a complexity-benefit tradeoff by comparing them to
simple aggregation of events. We use three clinical tasks involving time-varying
structured Electronic Health Records (EHRs) and increasingly clinically relevant
problems. We show that these benchmarking tasks display heterogeneous predictive
difficulties. We introduce a simple aggregation of static embeddings
--transferred from national claims and publicly available--, showing that it
outperforms transformer-based models on simple tasks with medium sample sizes.
We highlight the sample and computing resource efficiency of these models.
Finally, clinically relevant problems generally present a strong class
imbalance, which complicates models development and undermines their
performances. Further work is needed to understand if transformer-based models
perform well in these scenarios where the number of cases requires good sample
efficiency.

# Usage

See the [usage page on the documentation](https://soda.gitlabpages.inria.fr/medical_embeddings_transfer/usage.html)