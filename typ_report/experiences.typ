#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.
#show: project.with(
  title: "Event2vec, journal of experiments",
  authors: (
    (name: "Matthieu Doutreligne", email: "matt.dout@gmail.com", affiliation: "HAS, Inria"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: "",
  date: "March 22, 2023",
)
#show link: underline

#show raw: box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#outline()

= Experiments

== Experimental setup overview

=== Tasks

Task 1 is LOS interpolation for every complete hospitalization. There is only 25,000 patients in the effective cohort, since 8,000 patients included do not have any events in the selected feature tables (icd10, procedures or, procedures).

Task 2 is next visit prediction (all hospitalizations: complete or incomplete).

#let image_dir = "../docs/source/_static/img/"
#let image_dir_cohort = image_dir + "cohort/" 
#let image_dir_experiment = image_dir + "experiences/" 

#grid(
  columns: (1fr, 1fr),
  align(center + horizon)[
#figure(
  image({image_dir_cohort + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3.svg"}, width: 100%),
  caption: [
    Inclusion criteria for T:LOS task.
  ],
)
  ],
   align(center+horizon)[
#figure(
  image({image_dir_cohort + "icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01.svg"}, width: 100%),
  caption: [
    Inclusion criteria for T2:Prognosis task.
  ],
)
  ],
)

=== Evaluation setups

The first evaluation setup focus on the statistical efficiency gains of these
embedding models. We fix a test set size (0.3 of total task population), then sample training sets of increasing sizes.

=== Features

By default, the features are the following:
- Billing codes (ICD-10)
- Procedure codes (CCAM)
- Administration drugs (ATC7)

We also add static features corresponding to the index stay of the task (T1=target stay, T2=first included stay): age, gender, admission reason, discharge destination, type and value. Finally, inclusion dates have been enriched to include the day of the week, and the month. The moment of the day is not included, since it is 22:00 or 23:00 for all visit starts.

A decay over the event is added to the features to include temporality: it is a decreasing exponential weight with half-life time depending of the task.

== Task 1: LOS interpolation

=== Efficiency of the embedding models for T1:LOS 

For this task, the exponential decay has been set to 7 days, focusing on events in the short term. 

#figure(
  image(
    image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_all_three_estimators/roc_auc_score_performances.png"
  ),
  caption: [
    Performance of the embedding models for T1:LOS task. The study vocabulary is composed of 4734 codes, from which 4265 are found in SNDS embeddings and 2100 in cui2vec embeddings.
  ],
)<t1_s1>


If I restrict all models to the vocabulary of 2100 codes that I successfully mapped from UMLS cui identifiers to the french vocabulary, I get the following results. THis gives a slight advantage to locally trained and cui2vec embeddings compared to SNDS where only 1987 study codes are found. For HGB, the performances are very similar between embedding featurizers (wo dimension reduction) for boosting. Forests have slightly worse performances and favor local embeddings.

#figure(
  image(
    image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_all_three_estimators_restricted_to_cui_voc/roc_auc_score_performances.png"
  ),
  caption: [
    Performance of the embedding models for T1:LOS task. Vocabulary has been restricted to 2100 common codes between UMLS cui identifiers and the study vocabulary.
  ],
)<t1_s1_restricted_voc>


=== Transfer 

==== Decay 7 

#figure(
  image(image_dir_experiment + "transfer__complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_3_models_decay7/roc_auc_score_performances.png", width: 100%),
  caption: [
    T1:LOS task, transfer between hospitals: The best model is the embeddings retrained from scratch on APHP, concordant with the efficiency setup. The SNDS embeddings with forest and boosting seems to close the gap compared to the efficiency setup.
  ],
)

Transfering with a restriction to the 2100 codes of cui2vec vocabulary. 

#figure(
  image(image_dir_experiment + "transfer__complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_restricted_to_cui_voc/roc_auc_score_performances.png", width: 100%),
  caption: [
    T1:LOS task, transfer between hospitals with 2100 codes only: The local, SNDS or cuiv2vec embeddings with forest and boosting are all equivalent in forests and boosting. For logistic regression, the local embeddings remain the bests.
  ],
)




Looking at the brier does not change the conclusion. 

#figure(
  image(
    image_dir_experiment + "transfer__complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_3_models_decay7/brier_score_loss_performances.png"
  ),
  caption: [
    T1:LOS task, transfer between hospitals: There is no clear advantage in term of Brier score either for SNDS embeddings, beside a small advantage for embeddings+SVD. NB: These are even better results for brier score than the one obtained within the efficiency setup.
  ],
)


==== Decay is 30 (error on my part)

With the logistic regression, the SNDS embeddings transfer not very well. However, with forest or hgb, we have similar performances to the embeddings retrained from scratch on APHP. Forcing a distribution shift does not seems to hinder performances. For the Forests estimator, we got the best performances with a decay of 1 day. It changes the performances in the same direction but with different optimal value for each featurizer.

#figure(
  image(image_dir_experiment + "transfer__complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_3_models_decay30/roc_auc_score_performances.png", width: 100%),
)

=== Ablation study

==== What effect of the decay parameter ? 

We clearly see that the decay parameter should at least be crossvalidated on the task, since it has a huge impact on the performances.

#figure(
  image(image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_ablation_decay_RF/roc_auc_score_performances.png", width: 100%),
  caption: [
    Ablation study on decay parameter for T1:LOS task.
  ],
)


==== What effect of the demographics ?

No effect for logistic regression (#ref(<los_ablation_demographics_lr>)) expect a slight effect for the Count encoding + SVD.featurizer. However, for random forests (#ref(<los_ablation_demographics_rf>)), the demographics improve the SNDS embedding method, bringing it to the same level as the embeddings retrained from scratch.

This suggests that simple demographics are easilly captured by concept embeddings, something already reported for attention based trajectory embeddings li2020behrt.


#figure(
  image(image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_ablation_static_features_LR/roc_auc_score_performances.png", width: 100%),
  caption: [
    Ablation study on demographics with logistic regression for T1:LOS task.
  ],
)<los_ablation_demographics_lr>


#figure(
  image(image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_ablation_static_features_RF/roc_auc_score_performances.png", width: 100%),
  caption: [
    Ablation study on demographics with random forests for T1:LOS task.
  ],
)<los_ablation_demographics_rf>

Very little effect of the demographics for HGB for the embeddings featurizers.

#figure(
  image(image_dir_experiment + "complete_hospitalization_los__age_min_18__dates_2017-01-01_2022-06-01__task__length_of_stay_categorical@3_ablation_static_features_HGB/roc_auc_score_performances.png", width: 100%),
  caption: [
    Ablation study on demographics with random forests for T1:LOS task.
  ],
)<los_ablation_demographics_hgb>


== Task 2: Next visit prognosis

==== Performances to expect from the #link("https://www.nature.com/articles/s41598-020-62922-y", "Behrt paper")

1.6m patients for Behrt pre-training (at least 5 visits and mapped to a icd10 or Read code). They reduce the vocabulary done to 301 codes for the whole analysis.
For the next visit codes predictive task, they keep 700K patients and  split them into 80/20 train/test, evaluated on ROC_AUC/APS (Average Precision Score). I am not entirely sure if they retrain from scracth or transfer the model. Looking at the #link("https://github.com/deepmedicine/BEHRT/blob/master/task/NextXVisit.ipynb", "next visit task notebook") and the #link("https://github.com/deepmedicine/BEHRT/blob/master/model/NextXVisit.py#L85", "NextVisit model"), it seems that it loads pretrained embeddings, so it might have a little bit of overfitting. However, this is not the case for both Deepr and Retain, that are trained from scratch. IMHO it is an unfair comparison in their paper with these other models.

For next visit, they focus on above 60 having =>1% prevalences and reach 0.954/0.462 in AUROC/APS, which is sensibly the same as Deepr (0.943/0.360) or RETAIN (0.921/0.382) for AUROC but much better in term of APS. Looking at the 6 months prediction (their second task), there is still #link("https://www.nature.com/articles/s41598-020-62922-y/figures/6", "important hetegoreneities") in the performances depending of the Caliber chapter (nomenclature developped to merge UK codes). Overall, mental and behavioural disorders, then diseases of the circulatory system are well predicted. 

=== Efficiency of the embedding models for T2:Prognosis

Comparison of 7 vs 30 decay: 7 days decay is better for the circulatory (+5 ROC), endocrine (+2.5 ROC), mental (+3 ROC), respiratory (+3 ROC). If the effet of decay measured for RF on the LOS task are transferable, we could still gain some performances.

==== Logistic regression with 7 days decay  : 

#let prognosis_expe_dir_d7 = image_dir + "experiences/" + "icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_decay7/"
#let icd10_chapters_prev_sup_10 = ("21", "9", "4", "18", "2", "5", "11", "6", "14", "13", "10", "19")

#for c in icd10_chapters_prev_sup_10 [
  #figure(
    image(prognosis_expe_dir_d7 + "roc_auc_score__c_" + c + ".png", width: 100%),
  )
]

==== All models with 30 days decay 

There are decent peformances of in-domain embeddings, for some icd10 chapters such as circulatory system (ROC_AUC=0.80), or endocrine, nutritional and metabolic diseases (ROC_AUC=0.70), Symptoms, signs and abnormal clinical and labo findings (ROC_AUC=0.70) or mental disorders (ROC_AUC=0.75), Disease of the respiratory system (ROC_AUC=0.75). 

However these performances are never reached by the SNDS embeddings, that are seldom competitive with in-domain embeddings, and only in big samples setups (10,000 patients in train test). Sometimes, they strongly fail compared to in-domain training, as for the diseases of the circulatory system (ROC_AUC=0.70).

A general note: there is not signal ({ROC<=0.6}) in every task (eg. diseases of the musculoskeletal system.). Reassuringly, when there is signal, embeddings of the SNDS seems to largely outperform count encodings, but only at the price of big samples, compared to the in-domain embeddings.


#let prognosis_expe_dir_d30 = image_dir + "experiences/" + "icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_decay30/"

#for c in icd10_chapters_prev_sup_10 [
  #figure(
    image(prognosis_expe_dir_d30 + "roc_auc_score__c_" + c + ".png", width: 100%),
  )
]

=== Transfer of the embeddings for T2:Prognosis

The SNDS embeddings transfer performances depends on the estimator. 

For forest, they are systematically competitive or better than local embeddings. Sometimes the improvement over locals is striking in small sample regimes (factor influencing health), other time in big sample regimes (neoplams + RF, disease of the nervous system + RF, digestive system + RF) suggesting real predictive power beyon mere extrapolation of linked codes, diseases of the nervous system.
However, for logistic regression and boosting, they perform poorly compared to local embeddings. 

For each ICD10 chapter, the best performances are almost the same between local embeddings + boosting or SNDS + forests for circulatory system, endocrine diseases, and respiratory system. Expect for neoplasms, digestive systems and nervous system where SNDS + forests is better. For mental and symptoms, the best performances are obtained with local + forests.

Very interstingly, the performances of the SNDS embeddings outperforms the performances the perforamnces in the efficiency setup in several cases: fator influencing health (+5 ROC), neoplasm (+2.5 ROC), digestive (+2.5 ROC), nervous system (+5 ROC), respiratory (+2.5 ROC). In every case, SNDS stands out significantly from the local embeddings (except for the respiratory system).  


#let prognosis_expe_dir_transfer_d7 = image_dir + "experiences/" + "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_3_models_decay7/"
#let icd10_chapters_prev_sup_10_transfer = ("21", "9", "4", "18", "2", "5", "11", "6", "13", "10", "7")

#for c in icd10_chapters_prev_sup_10_transfer [
  #figure(
    image(prognosis_expe_dir_transfer_d7 + "roc_auc_score__c_" + c + ".png", width: 100%),
  )
]


=== Transfer of the embeddings for T2:Prognosis, restricting to 2100 cui2vec codes.

NB: due to time constraint, the HGB with counts did not ran. 

cui2vec is less interesting than the local or SNDS embeddings in this setup with better performances than the local embeddings, only for the strange case of disease of the eye, where counts seems to be sufficient to draw robust conclusions.

#let prognosis_expe_dir_transfer_d7 = image_dir + "experiences/" + "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_restricted_to_cui_voc/"
#let icd10_chapters_prev_sup_10_transfer = ("21", "9", "4", "18", "2", "5", "11", "6", "13", "10", "7")

#for c in icd10_chapters_prev_sup_10_transfer [
  #figure(
    image(prognosis_expe_dir_transfer_d7 + "roc_auc_score__c_" + c + ".png", width: 100%),
  )
]

=== Cehr-Bert attention model 

==== Instantiating the model on the APHP sample

I forked the #link("https://github.com/cumc-dbmi/cehr-bert", "CEHR-BERT github") of
cehr-bert an attention-based model, that added visit type prediction, to the
BEHRT model. My reasonning was that it is OMOP based so maybe more easy to use
on APHP data, furthermore they included other domain than cim10 in the models
(something not done by BEHRT).   

I rely on train/test split by hospital id for the pretraining and the fine-tuning. 

- I Initiated a #link("https://github.com/strayMat/cehr-bert/blob/eds_adaptation/eds_notes.md", "eds readme.md") for adaptation of cehr-bert to APHP-EDS data. Need a bit of cleaning for newcomers.


*Key points:*

- *pretraining:* it was quite easy to launch a pretraining on the APHP data (5h of work maximum to adapt the code). I only ran the pretraining on the training data of icd10chapter prognosis task ie 8297 persons, 622 000 events. I ran 2 epochs in 40 minutes on 10 CPUs (following their tutorial HPs). On one physical GPU (device: 0, name: Tesla T4, pci bus id: 0000:12:00.0, compute capability: 7.5) with 14GB, it tooks 6min.

- *finetuning:* Their code for cohort generation is hard to read and follow, so I added a code snippet in cehr-bert fork to create a binary task from an existing CohortEvent object which is basically two dataframes : person and event where person has a target column called y and events contain the eligible events for prediction (should be only after followup start if predictive task). 
      
I finetuned on the same cohort of 8297 persons, 622 000 events reducing the multi-label tasks to next visit icd10 chapters only; 9-circulatory system and 2-neoplasms. The performances for the best concept embeddings models are respectively 77.5\% and 65\%. 
The performances reported for next 6 month prediction with Behrt are: 2) Neoplasms: APS=0.20/0.58, ROC_AUC=0.86/0.97 for primary malignancy other skin and primary malignancy prostates (prevalence around 1\%); 9) Disease of the circulatory system: mean APS=0.37, mean AUROC=0.88, mean ratio=\% over 6 caliber chapters.

I ran 10 epochs respictevely in 1h45 on 5 CPUs or in  15 minutes on one T4 GPU (following their tutorial HPs). Their final predictor is a BiLstm model on top of the bert pretrained model:


- *Evaluation*: I transfered the learned model on the test cohort of 1714 patients
 79 946 events. I got excellent results for this task. I doubled check, but it seems that the model never saw the test example.

#figure(
  caption: "Fine tuning cehr-bert model works gives consistent results with the BEHRT paper.",
  table(
    columns: 6,
    align: center,
    [*ICD10 chapter*],[*recall*], [*precision*], 	[*f1-score*], [*pr_auc*], [*roc_auc*],
    [2: Neoplasm], [0.583658],[0.761421], [0.660793], [0.670023], [0.867384],
    [9: circulatory system], [0.851282],[ 0.935211], [0.891275], [0.947144], [0.97181],
  )
)

==== Exhaustive evaluation with full training data for icd10 next chapter prediction

I struggled a little bit on putting the model on GPU. I had an error mentionning #link("https://gitlab.inria.fr/soda/matthieu_doutreligne/-/issues/193","symbolic tensors"). I solved it by forcing tensorflow==2.3.0, cudnn==7.6.5 and numpy==1.18.5. In the process, I retrained a preprocessing model from the beginning on GPU. 

I ran this later model on the 21 chapters and got the following results.

#let results = csv(image_dir_experiment + "cehr-bert_beginnings/icd10_progognis_gpu_pretrain_21_chapters.csv")

#table(
  columns: 8,
  align: center,
  ..results.flatten(),
)

There is a strange result on chapter 2, less performant with gpu-pretrain (AUROC=[0.62,0.64], and recall=precision=0) on two distinct finetuning runs compared to the results obtained with the cpu-train (AUROC=[0.86, 0.85], recall=precision=0.6).
I am not sure what is the reason for this difference, besides different network initialization.
I tried a long run for chapter 2 (20 epochs) using the gpu-pretrain to see if I recover the results from the cpu-pretrain. I got an AUC of 0.93. However, after having fix the seeds, I am not able to recover this result.

I ran 5 different seeds with epoch=10 on target 2 to see how much variability I get. Pushing to 20 epochs gives almost the same results (mean ROC_AUC=0.6163). 

#let icd10_prognosis_gpu_pretrain_chapter2_p10_rs5 = csv(image_dir_experiment + "cehr-bert_beginnings/icd10_prognosis_gpu_pretrain_chapter2_p10_rs5.csv")
#figure(
  caption: "Forcing different seeds inside the Bert evaluator gives consistent bad results for the Neoplasm chapter.",
  table(
    columns: 6,
    align: center,
    ..icd10_prognosis_gpu_pretrain_chapter2_p10_rs5.flatten()
  )
)

When doing the same with a different pretrained model, I got better results. Recall that the only change was the pretrain model seed (and the fact that this model has been trained on cpu, but that should not impact performances).

#let icd10_prognosis_gpu_pretrain_chapter2_p10_rs5_cpu = csv(image_dir_experiment + "cehr-bert_beginnings/icd10_prognosis_gpu_pretrain_chapter2_p10_rs5_cpu.csv")
#figure(
  caption: "Using a different seed for the pretrain model, I got much better
  results for the Neoplasm chapter.", table(
    columns: 6,
    align: center,
    ..icd10_prognosis_gpu_pretrain_chapter2_p10_rs5_cpu.flatten()
  )
)


==== Evaluating the full CEHR-BERT pipeline (pretraining, finetuning) 

===== Comparaison with cui2vec favorables codes (information leakage)

I compared only to best performing methods with cui2vec favorables codes (restriction to 2100 codes). 

#figure(
 image("../docs/source/_static/img/experiences/transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_restricted_to_cui_voc_cehr_bert/roc_auc_score__c_macro.png"),
 caption: "The CEHR-BERT pipeline gives better results than the best performing methods (restriction to 2100 codes favorables to cui2vec).",
)

I then moved to all codes (4734 ICD10, procedure and drugs). 
Before the validation split by hospital ID for cehr_bert:

#let prognosis_before_hospital_split = image_dir + "experiences/" +"transfert_icd10_prognosis_before_hospital_validation_split/"

#figure(
  image(
    prognosis_before_hospital_split + "roc_auc_score__c_macro.png",
  ),
  caption: "Macro AUC before validation split by hospital for CEHR-BERT."
)

#figure(
  image(
    prognosis_before_hospital_split + "roc_auc_score__c_micro.png",
  ),
  caption: "Micro AUC before validation split by hospital for CEHR-BERT."
)


A mini-sensitivity analysis of the importance of the decay parameter shows that it matters quite a lote with a 3 to 6% gain in ROC-AUC at the micro level. The largest differences are for logistic regression and in-domain train embeddings (comparison of red to orange dotted curves).

#let prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split_w_leakage = image_dir + "experiences/" + "transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01_hospital_split_cehr_bert/"

#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split_w_leakage + "sensitivity-analysis-decay-for-icd10-transfer.png"),
  caption: "Adding a simple decay to the event improves performances at the micro and macro level from 3 to 6 points in ROC_AUC."
  )

  

- with box plots:

#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split_w_leakage + "prevalence_results__est_random_forests_boxplot.png"),
  caption: "Except for CEHR-BERT, the performances of all featurizers chained with random forest estimators are independant from the chapter prevalence. Below 15% of prevalences, random forets manage to extract information from pretrained concept embedding methods and outperform CEHR-BERT."
)


#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split_w_leakage + "prevalence_results__est_ridge_boxplot.png"),
  caption: "The performances of all featurizers chained with logistic regression benefit from higher chapter prevalences. The benefit are higher for CEHR-BERT."
)


Training time is also important to consider: 

#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split_w_leakage + "training_testing_time_per_chapter.png"),
  caption: "The training time (pretraining and finetuning) of CEHR-BERT is much higher than the other featurizers."
  )



===== Final experience with all codes and hospital transfer (without information leakage)


#let prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split = image_dir+"experiences/"+"transfer__icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__hospital_split_cehr_bert/"

#let icd10_chapters_prev_sup_10_transfer_cehr_bert = ("micro","macro", "21", "9", "4", "18", "2", "5", "6","11", "14", "10", "13", "19")


The micro and macro average results as well as results on the 21 chapters are as follow. 

The Naive Baseline (previous stay) is stronger than all embedding models for almost all chapters. It is beaten only for "infectious and parastic diseases" (not by a large margin), external causes of morbidity, and pregnancy (where count are doing great, which probably indicate that some procedures markers are strongly predictive of pregnancy related codes), and diseases of the ear and mastoid process (only good performances of embeddings).

SNDS Embeddings are doing better than every other methods on macro and micro average. Beam embeddings performed the worst. Train embeddings are not better than count featurizer + random forests.
I did not retry boosting.

**Perspectives**: These results are a bit deceiving and underline the need to change task:

- Retry rehospitalization ?
- Wait for progress from Julie on rehospit
- Try to add more codes (eg. prescribed drugs, until now, there are only administrated drugs.)

Results:

#for c in icd10_chapters_prev_sup_10_transfer_cehr_bert [
  #figure(  
    image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split + "roc_auc_score__c_" + c + ".png", width: 100%),
  )
]

When we look at the difference of performances depending on the prevalences, we have the following results for the 21 chapters. Different versions of this figure :

#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split + "prevalence_results__est_random_forests_scatter_xlog.png"),
  caption: "Except for CEHR-BERT, the performances of all featurizers chained with random forest estimators are independant from the chapter prevalence. Below 15% of prevalences, random forets manage to extract information from pretrained concept embedding methods and outperform CEHR-BERT."
)


#figure(
  image(prognosis_expe_dir_transfer_d7_cehr_bert_hosp_split + "prevalence_results__est_ridge_scatter_xlog.png"),
  caption: "The performances of all featurizers chained with logistic regression benefit from higher chapter prevalences. The benefit are higher for CEHR-BERT."
)

== Task 3: Predicting MACE complication in incident patients


=== Task definition

The cohort are the patients with a hospitalization (0 days and more ie.
incomplete and complete) visits and at least 2 visits and at most 7 visits. The index visit is defined either as the first, last or radom visit respecting the following inclusion criteria: being during the study period: 01-01-2017/01-06-2022, having at least 360 days of followup (before 01-06-2022), no in-hospital mortality, aged over 18 at admission, with at least one billing code.  

The task definition adds one exclusion criteria: No MACE during the first visit (incident MACE in the database). Then it searches for a MACE code during the 360 days of followup after the end of the index visit.

I found 20308 / 429860 = 4.72% prevalence of MACE in this cohort when studying the 01-01-2017/01-06-2022 period and the first visit as index. However this ratio drops to less than 1% when choosing the last visit as index. 

After looking at the occurences of MACE codes during the whole study period, I saw distributional shift with respect to time, ie. a drop in MACE codes after the end of 2021 and before 2018. 

To stay on a stable regime, I restricted the study period to 2018-2021. For random index viit, it gives 5550 / 221455 = 2.51% prevalence of MACE in the cohort. For last index visit, it is only 1.05%. For first index visit, it is XX%.

 
Enhancement: 
- what to do with dead patients ? Target not considered yet during followup.
- Add a blank period (how much). What to do with patient meeting the criteria during the blank period ?
- Our patients are not perfectly incidents since we only select the MACE after the followup period. It might make more sense to remove patients having MACE before the start of the followup period (defined as the end of the index visit).

I moved away from the 200k omop samples to the 2M samples from the diabetes project. The reason are "MACE" is a common complication of diabetes so it is good application in addition the diabetics foot and it has sufficient sample size compared to the 200K samples where I found only: 500/10534 patients = 4.74% of prevalence, so not enough sample for reliable testing of the model. 

==== Flowchart for random index visit and 2018-2021 period

#figure(
  image(
    image_dir+"/cohort/mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_random/flowchart_mace.png"
  )
)

==== Flowchart for last index visit and 2018-2021 period

#figure(
  image(
    image_dir+"/cohort/mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_last/flowchart_mace.png"
  )
)


==== Flowchart for first index visit and 2018-2021 period

/*#figure(
  image(
    image_dir+"/cohort/mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_first/flowchart_mace.png"
  )
)
*/

=== Results for 2017-2022 period

=== First index visit

#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2017_2022__task__MACE@360_3_estimators/roc_auc_score__performances_hist_gradient_boosting.png"),
  caption: "Results for MACE and HGB."
  )

#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2017_2022__task__MACE@360_3_estimators/roc_auc_score__performances_random_forests.png"),
  caption: "Results for MACE and forest."
  )


#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2017_2022__task__MACE@360_3_estimators/roc_auc_score__performances_ridge.png"),
  caption: "Results for MACE and ridge."
  )


=== Results for 2018-2021 period (bad end of visits for incomplete hospitalizations)

==== Random index visit 

#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_random_3_estimators/roc_auc_score__performances_hist_gradient_boosting.png"),
  caption: "Results for MACE and HGB."
  )

#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_random_3_estimators/roc_auc_score__performances_random_forests.png"),
  caption: "Results for MACE and forest."
  )


#figure(
  image(image_dir_experiment + "transfer__mace__age_min_18__dates_2018_2021__task__MACE@360__index_visit_random_3_estimators/roc_auc_score__performances_ridge.png"),
  caption: "Results for MACE and ridge."
  )


=== Results for 2018-2021 period (good end of visits for incomplete hospitalizations)

There is a strange drop in prevalence between test set (per hospitals, 1.30%) and train set (3.05%). 

This seems to be due to MACE events arriving on the same day as discharge from index hospitalization present in the train but not in the test. I'm having trouble understanding this difference, but it completely rots the algorithm's performance, especially on the average_precision_score.


= New experiments by time split

== LOS

- 2023-08-10: 
 - **Testing the los experiment framework:** subtrain grid=0.01, grid_decays = [[0], [0, 1], [0, 7], [0, 30], [0, 90]]. 
   - count featurizer: It takes 2-3 min to run over the 10 iterations of the RS.
   - All featurizers are passing. I prepare a big experience.
 - **I ran the full ML models on slurm:** train_grid=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1], grid_decays=[[0], [0, 1], [0, 7], [0, 30], [0, 90]], 10 iterations by random search. It fails for the snds2vec because of dimension mismatch. I suspect that it is the fault of the caching. I will remove caching and replace it by pretrained models that have been trained on the train sets (preambule of the prediction script if the embedding do not exist). 

- 2023-08-11: Focusing on making ok all ML models (not c-bert). 
 - Los is debugged and running
 - prognosis is debugged and running
 - mace is in test phase (30K patients only):
   - passed: subtrain_grid[0.01], models=ridge+forest, all featurizers, n_min_events=100.
   - passed : subtrain_grid[1], models=ridge+forest, with n_min_events=100, count featurizer. I see 7.5GB of memory for RF.
   - passed : subtrain_grid[1], models=ridge+forest, with n_min_events=10, count featurizer. I see 12GB of memory for LR,  but passing, RF is fine with 11GB. 
 - mace is in launch phase: 
   - failed: subtrain_grid[0.1, 1], n_min_events=10, all featurizers. It did  not went well. Especially it crashed for the count featurizer (and LR). 
   - launched: subtrain_grid[0.1, 0.5, 1], n_min_events=50, all featurizers. 

#bibliography("biblio.bib") 
