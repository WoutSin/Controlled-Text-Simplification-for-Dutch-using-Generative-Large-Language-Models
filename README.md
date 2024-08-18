# Dissertation: Controlled Text Simplification for Dutch using Generative Large Language Models 

The code in this repository is used to generate and evaluate controlled text simplification for Dutch using Generative Large Language Models (LLMs). Generation is achieved through an N-shot prompting methodology for controlled text simplification, utilizing the open-source multilingual LLAMA 3 model, as well as the open-source monolingual GEITje 7B ultra model. 

Subsequently, simplified outputs are evaluated with regards to the level of simplification by means of an Automatic Readability Assessment (ARA) model. I compared the performance of a more traditional feature-based approach and a neural approach using the Dutch RobBERT model for the development of a Dutch ARA model.

In the original study, simplified outputs were also assessed through manual error checking and human evaluation by Flemish teachers. The complete error analysis, along with the Dutch surveys employed for this evaluation, are also provided in the repository.

## Execution Order of Scripts

The scripts should be run in the following order:

1. **optional:** run the **Statistical ARA Model** > `Data_Histogram_Plot.py` script to visualize any set of features across all readability levels
2. **Statistical ARA Model** > `Statistical_ARA_Model_Functions.py` for training of the statistical model and extraction of the train and test data IDs.
3. Copy the `train_ids.txt` and the `test_ids.txt` files to the **Neural ARA Model** folder.
5. **Neural ARA Model** > `Data_Extraction.py` for the extraction of the train and test sets based on the previously extracted IDs.
6. **Neural ARA Model** > `Neural_ARA_Model_Functions.py` for training of the neural model.
7. **optional:** run the **Neural ARA Model** > `Evaluation.py` script for additional evaluation metrics. 
8. Copy the `Neural_ARA_Model` with the saved model to the **N-shot Learning Text Simplification** folder.
9. Run either or both the `GEITje.py` and `LLAMA.py` scripts in the **N-shot Learning Text Simplification** folder.

## Additional Information on the Datasets

1. **Dataset used to train the ARA models**

    For the development of the ARA models, 1000 texts were sampled from two existing corpora:
        
    1. An **in-house corpus** that has been specifically compiled for **assessing readability of various Flemish educational levels**. (used for levels 0-3)
    2. The **Wablieft Corpus**, a collection of texts from the Belgian easy-to-read Wablieft newspaper (used for level 4)

2. **Dataset used to generate simplifications using the Generative LLMs**

    As input to the Generative LLMs, a **corpus of 50 Dutch texts**, originating from either textbooks or a teacher co-creation platform was compiled. 

## In-depth overview on the Scripts

1. **Data_Histogram_Plot.py**
    - Reads and labels T-scan data on the different readability levels
    - Combines data from multiple sheets into a single DataFrame
    - Generates histograms and line plots for specified features across the readability levels
    - Saves the generated plots as image files in a designated output directory
    - Provides a visual representation of feature distributions for different readability levels

2. **Statistical_ARA_Model_Functions.py**:
    - Imports and preprocesses text data from an Excel file, performing feature selection using SelectKBest.
    - Splits the preprocessed data into training and test sets.
    - Trains and evaluates multiple machine learning models (Logistic Regression, Decision Trees, Random Forests, SVMs, Gradient Boosting, KNN, Naive Bayes) on the training data using cross-validation.
    - Finds the optimal number of features (k) and the best model for each algorithm by iterating over different values of k and evaluating the models' performance.
    - Loads the best models, predicts on the test set, and reports the F1-scores and confusion matrices for each model.

3. **Data_Extraction.py**:
    - The `process_data` function reads relevant text files from the various readability levels, processes the content and stores the processed data in an Excel file.
    - The script takes four inputs: a list of filenames, a list of folder names, a list of labels, and the output file path.
    - Merges the text content from each file, chunks the text into smaller pieces if the word count exceeds a limit, and associates each processed text with a filename and label.
    - Creates a pandas DataFrame with columns for the filename, label, and processed text, and writes the DataFrame to an Excel file.

4. **Neural_ARA_Model_Functions.py**:
    - Reads text data and labels from Excel files for a text classification task.
    - Splits the training data into training and validation sets, and tokenizes the text.
    - Creates PyTorch datasets for training, validation, and testing.
    - Trains and evaluates a text classification model using the prepared datasets.
    - Saves the trained model for future use.

5. **Evaluation.py**
    - Loads a pre-trained neural model for Automatic Readability Assessment (ARA)
    - Prepares test data by tokenizing and encoding it using a Dutch RoBERTa model
    - Sets up a Trainer with the loaded model and predefined training arguments
    - Performs predictions on the test set using the trained model
    - Calculates and prints evaluation metrics including confusion matrix, accuracy, precision, recall, and F1 score

6. **GEITje.py**:
    - Loads Dutch texts from an Excel file and pre-trained language models (ARA and GEITje).
    - Iterates over 50 texts from the corpus.
    - Generates simplified versions of the corpus texts at 4 different readability levels (0-3).
    - Predicts readability level of simplified texts using ARA model.
    - Stores original texts, simplified texts, target readability levels, and predicted levels in an Excel file.

7. **LLAMA.py**:
    - Loads Dutch texts from an Excel file and pre-trained language models (ARA and LLAMA 3).
    - Iterates over 50 texts from the corpus.
    - Generates simplified versions of the corpus texts at 4 different readability levels (0-3).
    - Predicts readability level of simplified texts using ARA model.
    - Stores original texts, simplified texts, target readability levels, and predicted levels in an Excel file.

