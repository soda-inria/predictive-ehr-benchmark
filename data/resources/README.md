# Several resources for vocabulary mapping

- care_site_hiearchy_w_hospital_name : extracted from the eds-scikit[aphp] package using the [`get_care_site_hierarchy function`](https://aphp.github.io/eds-scikit/v0.1.5/datasets/care-site-hierarchy/)

- nabm2loinc_2020 : extracted the NABM to OMOP concepts mapping from [the interhop susana mapping tool](https://susana.interhop.org/) in june 2020. Then mapped to LOINC using the [athena](athena.ohdsi.org/) OMOP codes to loinc mapping. A notebook is available in the [notebooks folder](../../notebooks/V2_nabm_hierarchy.ipynb).

- LOINCFR_JeuDeValeurs_2023, the second sheets corresponds to nabm2loinc_2023 : extracted from [the bioloinc ANS website](https://bioloinc.fr/bioloinc/KB/#Group:uri=http://aphp.fr/Bioloinc/JDV_LOINC_Biologie;tab=props;) in april 2023.

- MRCONSO.RFF: UMLS base codes, extracted from the [UMLS website](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) in april 2023.

- [Ohdsi Athena](athena.ohdsi.org/) mapping between snomed and french nomenclatures (CCAM, ICD10, ATC, LOINC): extracted in april 2023.

# Derived resources 

- mapping_loinc2nabm_2020: there are multiple possible NABM codes for the same LOINC code. Looking at the corresponding NABM labels, they are often closely related, meaning there is some redundancy in the codes. This file makes an arbitrary choice to have only one NABM code per LOINC for embedding experiences.

- icd10_tree.csv : manually reconstructed tree of icd10 to get code chapters

- umls_big_subset : restriction of UMLS concepts to the vocabularies of interest : ["ICD10", "ATC", "LNC", "SNOMEDCT_US", "ICD10CM", "ICD10PCS"].