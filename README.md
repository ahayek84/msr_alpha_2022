**Team name**: ALPHA

**Student names**:
- Abdelrahman ALHayek
- Jesmin Aktar Mousumy
- AmeerAli Khan 

### Repository Instructions
0. For `model-type` have two choice `[random_forest,svc]` and `combine-labels` also we have two choices `[True,False]`. Use `combine-label` if you want to combine label 1-2 as Noice and 3-4 as Expert. 

1. Upgrade pip and install the required package: 


    `$pip install --upgrade pip `


    `$pip install -r requirement.txt`
2. Grid search the parameters: 


    `$cd msr_alpha_2022/process`


    `$python -m hyper_paramter_tuning --model_type <model-type> --combine_labels <combine-labels>`

3. Train the model and generate classification report.


    `$cd msr_alpha_2022/process`


    `$python -m main --model_type <model-type> --combine_labels <combine-labels>`

4. default values :
    - model_type : **random_forest**
    - combine_labels : **true**

### Please find the evaluation score below for each Models and combine-labels Type


`Random Forest with combine labels (3 Classes)`


![rf_combinelabels](https://user-images.githubusercontent.com/13449847/181070634-2cfc3ced-f04d-44f3-8cbf-e0618001d7e6.png)


`Random Forest with out combine labels (5 Classes)`


![rf_nocombinelabels](https://user-images.githubusercontent.com/13449847/181072222-7dbbcd91-8cdf-49a9-a192-a542b4e83f12.png)


`SVC with combine labels (3 Classes)`
![svc_combinelabels](https://user-images.githubusercontent.com/13449847/181070704-ef3b5ec6-462e-4d73-8b56-f005bcf1d706.png)


`SVC without combine labels (5 Classes)`
![svc_nocombinelabels](https://user-images.githubusercontent.com/13449847/181070725-a8c9ef8e-e99f-4858-a457-46d46b80f79b.png)

### Objective of reproduction with subsections as follows:
Description-This paper extends the expertise identification approaches, to the context of third-party software components (external libraries), using a supervised and unsupervised learning.
- **Input data**- The paper focus on three popular libraries: FACEBOOK/REACT (for building enriched Web interfaces), MONGODB/NODE-MONGODB (for accessing MongoDB databases), and SOCKETIO/SOCKET.IO (for realtime communication), to identify experts in these libraries.The selected features (per Github user) like number of commits, number of client projects a candidate expert has contributed to.
- **Output data**- The paper collects the data from mentioned libraries, and train classifiers like RandomForest and SVM to find the experts in these papers, the classifiers are evaluated by precision, recall, F-score.

### Findings of reproduction with subsections as follows:
- **Our Process is slightly different than paper**:,
1. Since Test dataset size is not clear we assume `20%` data for testing (before SMOTE)
2. It not clear if the model is train on all the datasource (react, mongodb, socketio) or trained on each datasource seperately. We trained our model on all the datasource (combination of REACT, SOCKETIO, MONGODB)
3. Author also uses grid search for hyper-parameter tuning but they have not published the choosen value. Hence we did our own hyper-parameter tuning. 
4. Author didnt publish which library they used for SMOTE processing, we used `smote_variants` API for SMOTE.
5. Paper only talks about few feature for filling N/A data and all other feature we have filled N/A with value -99999.
6. We are not sure if reported F-measure is macro or weighted average. Hence consired it to be weighted average.


- **Output delta**: 
1. Author reported maximum `F-measure of 0,56 (MONGODB)` for 3 class classifier. --> We reported `F-measure of 0.63 (MONGODB)`. `Delta of 0.07`
2. For 5 class classification, author only report score for `React` data because other dataset don't have sufficient data. `Precision ranges from 0.0 (Novice 2, SVM)` to `0.50 (Expert 4, Random Forest)`. `F-measure is 0.24 (Random Forest) and 0.15 (SVM)`. --> We report `precision 0.0 (Novice 1, RF) to 1.0 (Expert 5, RF)`. `F-measure of 0.38 (RF) and 0.34 (SVC)`. Over all there is `delta of 0.5 for precision and 0.14-019 for F-measure`
3. For 3 class classification, author report `precision` results are greater for experts than for novice, both for `REACT (0.65 vs 0.14)` and `MONGODB (0.61 vs 0.60)`, while socketio has the highest precision for `novices (0.52)`. --> We report higher precision for Novice than intermediate `(1.0 vs 0.6, MONGODB)`. For React precision of expert is higher than `Novice (0.61 vs 0)` for socketio novice have highest `precision (0.43)`. `Delta of  (-0.04 vs -0.14) for react. For MONGODB -0.39 and for socketio -0.09`
4. For 3 class classification, author report `recall`, range from `0.09 (REACT, novices)` to `0.83(REACT, experts)`. --> We report recall of `0.0 (REACT, Novice)` and `0.9 (REACT, Expert)`. `Delta of -0.09 to 0.07`
5. For 3 class classification, author report `F-measure` is `0.36 (REACT)`, `0.56 (MONGODB)` and `0.42 (SOCKETIO)`. --> We report F-measure of `0.49 (REACT)`, `0.63 (MONGODB)`, `0.3 (SOCKETIO)`. `Delta of 0.13, 0.07, -0.19 respectively.` 
6. For Detailed report please check the images above. 
### Implementation of reproduction with subsections as follows:
- **Hardware requirements** : Google colab 
- **Software requirements** :sklearn - smote_varients - pandas - numpy and for further information check requirments.txt
- **Validation**: We create train test split of 80 , 20 ration , and used the test data for validation , we are generating complete classification report which contains precision,recall, f score for each data source 
- **Data**: Original training data size is 460 and after apply smote the training size increased to 930 , while the testing size 115 rows for all libraries 