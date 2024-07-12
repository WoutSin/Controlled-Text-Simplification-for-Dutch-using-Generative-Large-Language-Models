import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    make_scorer,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)


def import_and_preprocess(path, k=100):
    """
    Imports and preprocesses T-scan data from an Excel file, performing feature selection using SelectKBest.

    Args:
        path (str): The path to the Excel file containing the T-scan data.
        k (int): The number of features to select using SelectKBest.

    Returns:
        X_new (pd.DataFrame): The preprocessed data with selected features.
        max_features(int): The maximum number of features in the original data.
        selected_features (list): The names of the selected features.
        label_map (dict): A mapping of readability level indices to their corresponding names.
    """

    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names
    label_map = {i + 1: name for i, name in enumerate(sheet_names)}

    dfs = []
    for i, sheet in enumerate(sheet_names):
        df = xl.parse(sheet)
        df["readability_level"] = i + 1
        dfs.append(df)
    df_combined = pd.concat(dfs)

    missing_cols = df_combined.columns[df_combined.isnull().any()]
    df_combined.drop(columns=missing_cols, inplace=True)

    X = df_combined.drop(
        columns=[
            "Inputfile",
            "readability_level",
            "Par_per_doc",
            "Zin_per_doc",
            "Word_per_doc",
        ]
    )

    max_features = len(X.columns)

    Y = df_combined["readability_level"]

    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, Y)

    selected_features = X.columns[selector.get_support()]

    X_new = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])

    X_new.insert(0, "Inputfile", df_combined["Inputfile"].values)
    X_new.insert(1, "readability_level", Y.values)

    return X_new, max_features, selected_features, label_map


def split_data(df, test_size=0.1):
    """
    Splits the input DataFrame `df` into training and test sets.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the features and target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the training and test sets for the features (`X_train`, `X_test`) and the target variable (`Y_train`, `Y_test`).
    """

    Y = df["readability_level"]

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=Y, random_state=27
    )

    train_df["Inputfile"].to_csv("train_ids.txt", index=False)
    test_df["Inputfile"].to_csv("test_ids.txt", index=False)

    X_train = train_df.drop(columns=["Inputfile", "readability_level"])
    X_test = test_df.drop(columns=["Inputfile", "readability_level"])
    Y_train = train_df["readability_level"]
    Y_test = test_df["readability_level"]

    return X_train, X_test, Y_train, Y_test


def train_and_evaluate_models(X_train, Y_train, k, cv=5):
    """
    Trains and evaluates multiple machine learning models on the provided data.

    Args:
        X (numpy.ndarray): The input data.
        Y (numpy.ndarray): The target labels.
        k (int): The number of features selected using SelectKBest.
        cv (int, optional): The number of cross-validation folds. Defaults to 5.

    Returns:
        dict: A dictionary mapping model names to a tuple containing the fold index, the average F1 score, and the trained model.
    """

    models = [
        ("Logistic_Regression", LogisticRegression(random_state=27, max_iter=500)),
        ("Decision_Tree", DecisionTreeClassifier(random_state=27)),
        ("Random_Forest", RandomForestClassifier(random_state=27)),
        ("SVM", SVC(random_state=27)),
        ("Gradient_Boosting", GradientBoostingClassifier(random_state=27)),
        ("KNN", KNeighborsClassifier()),
        ("Multinomial_Naive_Bayes", MultinomialNB()),
    ]

    model_f1_scores = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    for name, model in models:

        if name in ["Logistic_Regression", "SVM", "KNN"]:
            scores = cross_val_score(
                model,
                X_scaled,
                Y_train,
                cv=cv,
                scoring=make_scorer(f1_score, average="macro"),
            )

            f1_score_avg = scores.mean()

            model.fit(X_scaled, Y_train)

            model_f1_scores[name] = (k, f1_score_avg, model)

        else:
            scores = cross_val_score(
                model, X_train, Y_train, cv=cv, scoring=make_scorer(f1_score, average="macro")
            )

            f1_score_avg = scores.mean()

            model.fit(X_train, Y_train)

            model_f1_scores[name] = (k, f1_score_avg, model)

    return model_f1_scores


def find_optimal_k(path, start=10, interval=10):
    """
    Finds the optimal number of features (k) for the statistical ARA model by evaluating the model accuracy across a range of k values.

    Args:
        path (str): The file path to the dataset.
        start (int, optional): The starting value for the range of k values to test. Defaults to 10.
        interval (int, optional): The interval between k values to test. Defaults to 10.

    Returns:
        dict: A dictionary mapping model names to a list of tuples, where each tuple contains the model accuracy and the selected features for a given k value.
    """

    _, max_features, _, _ = import_and_preprocess(path)
    print(f"Maximum possible features: {max_features}")
    k_values = range(start, max_features + 1, interval)

    all_model_accuracies = defaultdict(list)

    for k in tqdm(k_values, desc="Progress", unit="feature_set"):
        data_frame, _, selected_features, _ = import_and_preprocess(path, k)
        X_train, _, Y_train, _ = split_data(data_frame)
        model_accuracies = train_and_evaluate_models(X_train, Y_train, k)

        for model, accuracy in model_accuracies.items():
            all_model_accuracies[model].append((accuracy, selected_features))

    return all_model_accuracies


def get_best_k(all_model_accuracies):
    """
    Finds the best k value and corresponding model for each model in the provided dictionary of model accuracies.

    Args:
        all_model_accuracies (dict): A dictionary where the keys are model names and the values are lists of tuples containing the k value, F1-score, and the model object.

    Returns:
        best_k_values (dict): A dictionary where the keys are model names and the values are the best ((k, F1-score, model), selected_features) tuples for each model.
    """
    best_k_values = {}
    for model, accuracies in all_model_accuracies.items():

        sorted_accuracies = sorted(accuracies, key=lambda x: x[0][1], reverse=True)

        best_k_values[model] = sorted_accuracies[0]
        print(
            f"{model}: best model found at k = {sorted_accuracies[0][0][0]} with F1-score = {sorted_accuracies[0][0][1]*100:.2f}%"
        )

        with open(f"Models/{model}_best_model.pkl", "wb") as f:
            pickle.dump(sorted_accuracies[0][0][2], f)

        with open(f"Models/{model}_best_k.pkl", "wb") as f:
            pickle.dump(sorted_accuracies[0][0][0], f)

        with open(f"Models/{model}_selected_features.txt", "w") as f:
            for feature in sorted_accuracies[0][1]:
                f.write(f"{feature}\n")

    return best_k_values


def load_models_and_predict(path, test_size=0.1):
    """
    Loads the pre-trained machine learning models, predicts on a test set, and prints the F1-score, accuracy, precision, recall and confusion matrix for each model.

    Args:
        path (str): The path to the dataset to be used for testing the models.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.

    Returns:
        test accuracies and confusion matrix of the best k models.
    """
    models = [
        "Logistic_Regression",
        "Decision_Tree",
        "Random_Forest",
        "SVM",
        "Gradient_Boosting",
        "KNN",
        "Multinomial_Naive_Bayes",
    ]

    with open("Results_Overview.txt", "w") as file:
        file.write("")

    with open(f"scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    for model in models:
        with open(f"Models/{model}_best_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        with open(f"Models/{model}_best_k.pkl", "rb") as f:
            best_k = pickle.load(f)

        df, _, _, label_map = import_and_preprocess(path, k=best_k)
        _, X_test, _, Y_test = split_data(df, test_size=test_size)

        if model in ["Logistic_Regression", "SVM", "KNN"]:
            X_scaled = scaler.transform(X_test)
            predictions = loaded_model.predict(X_scaled)
        else:
            predictions = loaded_model.predict(X_test)

        f1 = f1_score(Y_test, predictions, average="macro")
        accuracy = accuracy_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions, average="macro")
        recall = recall_score(Y_test, predictions, average="macro")

        with open("Results_Overview.txt", "a") as file:
            file.write(
                f"{model}: F1-score on test set = {f1*100:.2f}% ({best_k} features)\n"
            )
            file.write(
                f"{model}: Accuracy on test set = {accuracy*100:.2f}% ({best_k} features)\n"
            )
            file.write(
                f"{model}: Precision on test set = {precision*100:.2f}% ({best_k} features)\n"
            )
            file.write(
                f"{model}: Recall on test set = {recall*100:.2f}% ({best_k} features)\n"
            )
            file.write("\n")

        cm = confusion_matrix(Y_test, predictions)
        cm_df = pd.DataFrame(
            cm,
            index=[label_map[i] for i in loaded_model.classes_],
            columns=[label_map[i] for i in loaded_model.classes_],
        )

        with open("Results_Overview.txt", "a") as file:
            file.write(f"{model}: Confusion matrix:\n{cm_df}\n")
            file.write("\n")

        cm_df.to_csv(f"Models/{model}_confusion_matrix.csv")

        print(f"{model}: F1-score on test set = {f1*100:.2f}% ({best_k} features)")
        print(f"{model}: Accuracy on test set = {accuracy*100:.2f}% ({best_k} features)")
        print(f"{model}: Precision on test set = {precision*100:.2f}% ({best_k} features)")
        print(f"{model}: Recall on test set = {recall*100:.2f}% ({best_k} features)\n")
        print(f"{model}: Confusion matrix:\n{cm_df}\n")


def main():
    """
    Runs the main functionality of the statistical ARA model.

    This function performs the following steps:
    1. Finds the optimal value of k for the ARA model using the `find_optimal_k` function.
    2. Retrieves the best k models based on the model accuracies.
    3. Prints the test accuracies and confusion matrix of the best k models.
    """
    # Interval is set to 1000 to reduce the computation time (original = 10).
    all_model_accuracies = find_optimal_k(
        "Corpus_Selection.xlsx", start=10, interval=1000
    )
    best_k_models = get_best_k(all_model_accuracies)
    print("\nTest accuracies:")
    load_models_and_predict("Corpus_Selection.xlsx", test_size=0.1)


if __name__ == "__main__":
    main()
