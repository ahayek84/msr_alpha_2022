• Team name: ALPHA
Student names:
-Abdelrahman ALHayek
-Jesmin Aktar Mousumy
-AmeerAli Khan    
• Objective of reproduction with subsections as follows:
Description-This paper extends the expertise identification approaches, to the context of third-party software components (external libraries), using a supervised and unsupervised learning.
Input data- The paper focus on three popular libraries: FACEBOOK/REACT (for building enriched Web interfaces), MONGODB/NODE-MONGODB (for accessing MongoDB databases), and SOCKETIO/SOCKET.IO (for realtime communication), to identify experts in these libraries.The selected features (per Github user) like number of commits, number of client projects a candidate expert has contributed to.
Output data- The paper collects the data from mentioned libraries, and train classifiers like RandomForest and SVM to find the experts in these papers, the classifiers are evaluated by precision, recall, F-score.
• Findings of reproduction with subsections as follows:
• Process delta: How does your process differ from what’s described in the paper or its repo? (Why?)
In the paper they used the JavaScript libraries, the goal is to identify experts in these libraries based on their activity on GitHub.  REACT3 (a system for building enriched
Web interfaces), MONGODB/NODE-MONGODB4 (the official Node.js driver for MongoDB database server), and SOCKETIO/SOCKET.IO5 (a library for real-time communication) as well as in backend development and selected MONGODB/NODE-MONGODB, a persistence library, and SOCKETIO/SOCKET.IO. Our Process is slightly different than paper, in pre-processing step we use Mongodb, react, socketio for Read the data from data folder and data source. Moreover, we use Smote_varient library to oversample minority classes and null value filled with 0. We use MultiColumnLabelEncoder to Label none numerical data ["login","name","email","source"].
• Output delta: How does your output differ …? (What’s the significance of any differences observed?)

• Implementation of reproduction with subsections as follows:
• Hardware requirements : Google colab 
• Software requirements :sklearn - smote_varients - pandas - numpy and for further information check requirments.txt
• Validation: We create train test split of 80 , 20 ration , and used the test data for validation , we are generating complete classification report which contains precision,recall
f score for each data source 
• Data: Original training data size is 460 and after apply smote the training size increased to 930 , while the testing size 115 rows for all libraries 