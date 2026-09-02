*** Data Files ***
You must choose a task involving text analysis or image processing and select a dataset from Kaggle (choose something relatively simple).

For a text analysis/Natural Language Processing (NLP) task:
▪ You must select a text analysis dataset from Kaggle via this link:
https://www.kaggle.com/datasets?tags=13204-NLP

• For example, a dataset for classifying emails as spam or not-spam:
https://www.kaggle.com/datasets/team-ai/spam-text-messageclassification

For an image processing/computer vision task:
▪ You must select an image processing dataset from Kaggle via this link:
https://www.kaggle.com/datasets?tags=13207-Computer+Vision

▪ For example, a dataset of handwritten digits:
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

The following details must be entered in the shared Excel:
• Assignment type (text analysis/image processing)
• Learning type (learning type) – classification, regression or clustering
• Learning algorithms Implemented – the name of the algorithm you decided to implement.
• Dataset name – the name of the dataset, as it appears in Kaggle.
• Dataset URL – the link to the dataset in Kaggle must be entered.
• Video URL – the link to the video explaining the assignment must be entered (verify viewing permissions, for anyone who has a link). The link to YouTube or any other place where the video can be viewed (without downloading the video), for those who are not presenting in class
• Repository URL – the link to the repository where you uploaded the assignment must be entered, where the assignment can be viewed (verify viewing permissions, for anyone who has a link), without downloading it and the results of the run can also be viewed (the run of the assignment is your responsibility).


***Presentation of the assignment in class (bonus for submitting in class) or video of the assignment presentation ***
• The assignment will be most evaluated against the presentation of the assignment in class or the video you prepared.
• The presentation of the assignment in class or the video should be about 5 minutes long (try to be around that size)
• When presenting the assignment in class or in a video, you must present the material, assuming that the viewer does not know the material.
• At the beginning of the presentation in class or at the beginning of the video, you must introduce the students in the group.
• You must present the different parts of the assignment and its products, while showing the code and the output
and accompanying it with explanations, which show an understanding of what you did and the results.
• We expect you to share all group members in the video, as equally as possible.
• For those who submit the assignment via video, the focus of the video is on the code and products and there is no need to show
the team members.
• Failure to present in class, or failure to submit A video, or an unavailable or improper video, will incur a significant penalty in your grade.
• In summary: You must present in class or send a link to a clear, comprehensive, and inclusive video that shows and reviews your work and the deliverables in a comprehensive and clear manner and shows that you understood them and that they are working properly.
• Failure to present the assignment will incur a significant penalty

*** Code Notebook ***
• The content included in the assignment is explained in the Assignment Details section below.
• For those submitting a video, make sure that everyone who receives the link has access to the video.
• The notebook must also include the deliverables and visualizations, without downloading and running on the computer of the person who will review it.
• The notebook must be accompanied by notes and brief explanations that explain the work and deliverables.
• Please note – the assessment in the assignment is primarily on the process and not on the results in assessing the quality of the model.
• Failure to submit the solution notebook, or submitting an unavailable or incorrect solution notebook, will result in a significant penalty
in the grade.
• In conclusion: It is mandatory to submit a clear, comprehensive, and inclusive assignment notebook link that shows and reviews your work and the deliverables in a comprehensive and clear manner.
• Assignment without presentation will be significantly fined
• Assignment without a working link to the code will be significantly fined
• Assignment without submitting a proper link to the code and without presentation will not be reviewed, which is a shame

*** Additional instructions ***
1. For those submitting a video only (not for those presenting in class) - a link to the video (you will need to upload the video to YouTube, or another place on the Internet where the video can be viewed) in which you present and explain your work and the results.
https://youtu.be/kqtD5dpn9C8 - as shown in the example
▪ Please note, the video must be in good condition and available for viewing without the need to download.
2. A link to the code notebook that will open on the Azure / Colab / GitHub page of one of the participants, which includes the abbreviated student details, help you used in the AI ​​Chatbot implementation, the run, and comments.
https://git.new/FF6Dp2F - as shown in the example
▪ Please note, the code in the repository code should include explanations and output of the run for the different stages without the need to download.
3. The link to the dataset on Kaggle - the link to the dataset.
https://tinyurl.com/bdezpa8x -as appears in the example

*** What do you have to do in the assignment? ***
The assignment must run a supervised learning flow (classification learning or regression learning, of your choice, depending on the dataset).
• Explain all the steps you do in the video, when you present the code that you will upload
to Azure / Colab / GitHub (the repository).
• The score will also include a clear explanation, showing that you understood what you did

*** Part 1 – Introduction ***
At the beginning of the assignment (up to 5 points can be earned in this section).
The section includes the following sections: Student details, prompts, explanation of the dataset, loading the dataset
• Student details – at the beginning of the assignment, you will need to write down the first name and the first letter of the last name, plus the last 4 digits of the ID.
o In the video presentation, the names of the participants must be clearly displayed at the beginning
• Prompts in LLM AI or chatbots, additional aids A cell should be dedicated in which you will write the prompt you used in the chatbot AI, additional links you used and what was the purpose of using them –
This is allowed, but of course you must show understanding
o We expect this to also be addressed orally.
• Explanation of the learning problem and the dataset – a short summary of the problem and the dataset is required at the beginning of the submission file, one paragraph in length. You must explain in a little more detail about As stated in the video.
• Loading the dataset - the assignment should include loading the trainset and the testset
o Note - these datasets should not be split again into train and test.
o You must display the first 5 rows of each dataset


*** Quality index – (use of appropriate indices is measured separately, up to 10 points can be scored in this section) ***
• The following quality indices will be used both to evaluate the quality of the work and to compare it in experimenting with different features, in the quality assessment and bonus sections later (sections 5 and 6)
• In regression problems – the quality will be assessed according to the 2^r index.
• In classification problems (multi-class or with 2 classes, but without a central class) – 1f-average-macro
• In a binary classification problem (with only one central class) – 1f (only on the central class)

*** Part 2 – engineering feature (up to 35 points can be scored in this section) ***
• Metrics to experiment with are metrics that we learned for text analysis or image processing
• If you want to activate an additional engineering feature, do so in the framework of the extension, described in section 5.
• The - Engineering feature on 2-3 examples in train and test

*** Part 3 – Implementing a learning algorithm (up to 35 points can be earned in this section) ***
• You must implement a learning algorithm that you have registered for.
o Bonus (5 additional points), for relatively complicated algorithms (to be considered by the lecturers), such as ANN.
• You must allow the use of different hyperparameters – at least as taught in class
• You must allow a training function and a predict function
• You must go through the algorithm and explain it

*** Part 4 – Training – Running the flow according to the different parameters (up to 5 points can be earned in this section) ***
• If you chose to experiment with extensions (section 6), you must use the successful combination for the entire trainset (i.e. retrain), you must present the process that several examples (2-3 examples)
go through during the engineering feature.

*** Part 5 – Prediction and quality assessment The model on the test set (up to 10 points can be earned in this section) ***
• If you chose to experiment with extensions (section 6), you must use the successful combination for the entire test set, you must show the process that several examples (2-3 examples) go through during the
.feature engineering
• The prediction results of the first 5 classifications on the test set should be shown
• The quality of the model should be shown as described above (and using the special method for quality assessment, if you chose to perform such an extension).

*** Part 6 – Extension – Bonus Section (up to 45 points can be earned in this section) ***
** .6a. Experimenting with engineering features and hyper parameters (up to 25 points can be earned in this subsection) **
Displaying the results of this section:
• Display all permutations and the (average) results of the index in a dataframe
• Display the best permutation with the score of the best permutation in addition separately.
• Later in the training phase (section 4) – you must train the entire trainset with the most successful permutation
• In the testing phase (section 5) – you must perform the same preprocessing on the test examples as well.
• Experiment with different values ​​of the selected engineering feature.

grid-search k-fold cross-validation with the experiments Management of grid-search:
You must perform a grid search, in which you will test all permutations. That is, a Cartesian product of the engineering permutations feature, the different models created by the learning algorithms that you have chosen with the different hyperparameters.

K-fold cross validation:
• validation cross fold5- which will wrap all the parts in the experiments. Divide the trainset into 5-equal parts and each time use 4 parts for training and 1 for validation, perform each permutation from the search grid 5 times (1 for each fold) and test the index on the additional part (validation). You should calculate the average index of the index for the 5 folds for each permutation.
The extensions described below should be performed using validation cross fold-k search grid,
described above, in this section.

for feature engineerin:
• If you want to experiment with an engineering feature that is not taught in class, you should also do this with the engineering feature that is taught in class and perform some of the comparison against the engineering feature that is taught in class.
o In this case, you should expand the explanation in engineering feature Not taught in class
o You can also experiment with different options of the same engineering feature
• You are expected to perform specific and specific preprocessing required for the specific problem, dataset, and application type
• If you chose to experiment with different hyperparameters, add them to the Cartesian product to be tested
• The different experiments must be presented in a dataframe – and the average score in cross validation should be presented as mentioned in section 3a.
• Present the extension in the code notebook and orally
• Engineering feature experiment score
• Up to 5 points for implementing an experiment in each of the engineering features (a maximum of 20 points can be reached if experiments are performed with the engineering feature only)

for hyper parameter tuning:
• Experiment with different values ​​of selected hyperparameters.
• If you want to add a hyper parameter that was not taught in class, you should expand and explain it more
• If you chose to experiment with several hyperparameters, you should collect all possible permutations between the hyperparameters and their different values ​​
• If you also chose to experiment with different engineering features, add them to the Cartesian product to be tested
• The different experiments should be presented in a dataframe – and the average score in the cross validation
as mentioned in section 3a.
• Present the extension in the code notebook and orally
• Hyperparameters Experimentation Score
o Up to 5 points for implementing an experiment in each of the hyperparameters (10 points can be reached, at most, if experiments are performed with hyperparameters only)
• A total of 25 points can be scored in this section

** .6b. Additional Extensions - Special Data Adjustments (up to 10 points can be scored in this subsection) **
• Special Data Adjustments - Special data adjustments required for the specific problem, the
dataset, and the type of application, various treatments for imbalanced data that are mainly based on -under sampling, -oversampling, and the generation of synthetic examples.
• Present the extension in the code notebook and orally
• Score for this extension up to 10 points

** .6c. Additional Extensions - Explainability - Understanding the results, their accessibility, and their accessibility The features (up to 10 points can be scored in this subsection) **
• Understanding and analyzing the important data, using techniques embedded in the information held as a result of learning, which is returned with the model with the prediction, or with various techniques that analyze the predictions and features after the prediction, such as
.SHAP
• Present the extension in the code notebook and orally
• Score for this extension up to 10 points