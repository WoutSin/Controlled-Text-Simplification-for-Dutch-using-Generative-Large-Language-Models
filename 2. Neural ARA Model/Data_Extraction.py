from nltk.tokenize import sent_tokenize, word_tokenize
import os
import pandas as pd
import nltk

nltk.download("punkt")


def chunk_text(text, cutoff):
    """
    Chunks a given text into smaller pieces based on a word count cutoff.

    Args:
        text (str): The input text to be chunked.
        cutoff (int): The maximum number of tokens allowed in each chunk.

    Returns:
        str: The chunked text, with each chunk containing up to the specified number of tokens.
    """
    sentences = sent_tokenize(text)
    index = 0
    chunk = ""
    chunk_word_count = 0

    while index < len(sentences):
        sentence = sentences[index]
        if chunk_word_count + len(word_tokenize(sentence)) <= cutoff:
            chunk_word_count += len(word_tokenize(sentence))
            chunk += " " + sentence
            index += 1
        else:
            break

    return chunk


def print_longest_sequence(df):
    """
    Prints the length of the longest sequence in the provided DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text data.

    Returns:
        None
    """
    max_length = 0
    for text in df["text"]:
        length = len(word_tokenize(text))
        if length > max_length:
            max_length = length
    print("The length of the longest sequence is:", max_length)


def process_data(filenames, folder_names, labels, output_file):
    """
    Processes data files and generates an Excel file with the processed data.

    Args:
        filenames (list): A list of filenames to process.
        folder_names (list): A list of folder names containing the data files.
        labels (list): A list of labels corresponding to the readability level
        output_file (str): The path to the output Excel file.

    Returns:
        None
    """

    chunked = 0

    data = []
    for filename in filenames:
        for folder, label in zip(folder_names, labels):
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = [
                        line for line in file.readlines() if not line.startswith("###")
                    ]
                merged_text = "".join(text)

                word_count = len(word_tokenize(merged_text))
                cutoff = 512

                if word_count > cutoff:
                    chunked += 1

                chunk = (
                    chunk_text(merged_text, cutoff)
                    if word_count > cutoff
                    else merged_text
                )

                data.append([filename, label, chunk])

    df = pd.DataFrame(data, columns=["inputfile", "label", "text"])
    df.to_excel(output_file, index=False)

    label_counts = df["label"].value_counts()
    print(f"The number of items for each label are:")
    print(label_counts)
    print_longest_sequence(df)
    print(f"chunked: {chunked}")


def main():
    """
    Extracts text file paths and filenames from text files, and processes the data for training and testing.

    The `main()` function performs the following steps:
    1. Reads the `train_ids.txt` and `test_ids.txt` files to extract the text file paths for the training and test data.
    2. Extracts the filenames from the text file paths.
    3. Defines the folder names and labels for the data.
    4. Calls the `process_data()` function to process the training and test data, and save it to `train_data.xlsx` and `test_data.xlsx` files.
    """
    with open("train_ids.txt", "r", encoding="utf-8") as file:
        train_paths = [line.strip() for line in file.readlines() if line]

    with open("test_ids.txt", "r", encoding="utf-8") as file:
        test_paths = [line.strip() for line in file.readlines()]

    train_filenames = [path.split("/")[-1] for path in train_paths]
    test_filenames = [path.split("/")[-1] for path in test_paths]

    folder_names = ["vierdes", "zesdes", "secundair", "wablieft"]
    labels = [0, 1, 2, 3]

    process_data(train_filenames, folder_names, labels, "train_data.xlsx")
    process_data(test_filenames, folder_names, labels, "test_data.xlsx")

if __name__ == "__main__":
    main()
