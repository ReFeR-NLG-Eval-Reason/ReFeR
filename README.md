
### Implementation Details

- Currently, the codes for ReFeR are written for the ReFeR framework with the peer and AC models presented in the paper. They can be easily modified for other models as peer/AC.

#### config.py
- file for the keys for the API's used in the code. Please change the keys to the keys for the API's you are using.

#### AGIQA.py & ICQD.py
- Codes for running the AGIQA and ICQD datasets.
- Images for the dataset are to be saved in the images folder in the dataset folder with the same name as in the JSON file. Use the original test sets of AGIQA and ICQD.

#### Summeval.py
- Code for running ReFeR for the Summeval dataset.

#### Topicalchat.py
- Code for running ReFeR for the TopicalChat dataset.

#### Reasoning.py
- Code for running all the Reasoning datasets (AQuA, BBH_DU, CSQA, GSM8k).

#### NLG_results_eval.py
- ReFeR's results evaluation code for all the NLG datasets (Summeval, TopicalChat).

#### Reasoning_results_eval.py
- ReFeR's results evaluation code for all the Reasoning datasets (AQuA, BBH_DU, CSQA, GSM8k).

#### agiqa_results_eval.py
- ReFeR's results evaluation code for the AGIQA dataset.

#### icqd_results_eval.py
- ReFeR's results evaluation code for the ICQD dataset.

#### baseline_codes/
- Codes for all the baseline models used in the paper.
- All codes are used as instructed in the original papers.
- nlg_baseline_eval.py and reasoning_baseline_eval.py are used to evaluate the baseline models' performance on the NLG and Reasoning datasets respectively.
