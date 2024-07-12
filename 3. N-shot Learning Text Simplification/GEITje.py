import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline,
    Conversation,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def predict(text, tokenizer, ARA_model):
    """
    Predicts the label for the given text using the ARA model.

    Args:
        text (str): The input text to be predicted.

    Returns:
        int: The predicted label for the input text.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    # Create a DataLoader
    inputs = encoding["input_ids"], encoding["attention_mask"]

    # Predict
    ARA_model.eval()
    with torch.no_grad():
        outputs = ARA_model(*inputs)

    # Get the predicted label
    _, preds = torch.max(outputs.logits, dim=1)

    return preds.item()


def read_file(file_path):
    """
    Reads the contents of a file at the specified file path.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    """
    This function is the main entry point of the GEITje text simplification application. It performs the following steps:

    1. Loads an Excel file containing a corpus of Dutch texts.
    2. Loads a pre-trained ARA model and tokenizer.
    3. Loads a pre-trained GEITje conversational model.
    4. Iterates through the corpus texts, generating simplified versions of each text at different readability levels.
    5. Stores the original text, simplified text, readability level, and predicted readability level in a Pandas DataFrame.
    6. Writes the DataFrame to an Excel file named "Simplifications_GEITje.xlsx".
    """
    # Open the Excel file with the corpus
    df = pd.read_excel("Corpus_Nederlandse_Teksten.xlsx")

    # Cast the 'tekst' column to a list
    texts = df["Tekst"].tolist()

    # Load the saved ARA model and tokenizer
    ARA_model = AutoModelForSequenceClassification.from_pretrained("Neural_ARA_Model")
    tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2023-dutch-large")

    # load_in_8bit: lower precision but saves a lot of GPU memory
    # device_map=auto: loads the model across multiple GPUs
    chatbot = pipeline(
        "conversational",
        model="BramVanroy/GEITje-7B-ultra",
        model_kwargs={"load_in_8bit": True},
        device_map="auto",
    )

    # Set up parameters for pbar
    total_texts = len(texts)
    simplifications_per_text = 4    # max 4
    total_simplifications = total_texts * simplifications_per_text
    pbar = tqdm(total=total_simplifications, desc="Processing")

    # Define path to folder with system roles
    folder_path = "Readability_Levels_Synthetic_One_shot"

    start = 0
    end = 5

    # Generate simplifications and append relevant information to the dataframe
    while start <= 45:

        df = pd.DataFrame(
            columns=[
                "Prompt",
                "Original",
                "Readability_Level",
                "Simplification",
                "Predicted_Level",
            ]
        )

        for input_text in texts[start:end]:
            for i in range(simplifications_per_text):
                file_name = f"Readability_level_{i}.txt"
                file_path = os.path.join(folder_path, file_name)
                role_system = read_file(file_path)
                role_user = f"Vereenvoudig de volgende tekst naar niveau {i}: {input_text}"

                start_messages = [
                    {"role": "system", "content": role_system},
                    {"role": "user", "content": role_user},
                ]

                conversation = Conversation(start_messages)
                conversation = chatbot(conversation)
                response = conversation.messages[-1]["content"]

                input_prompt = f"{role_system}\n{role_user}"
                prediction = predict(response, tokenizer, ARA_model)

                # Add the response to the DataFrame
                new_row = pd.DataFrame(
                    {
                        "Prompt": [input_prompt],
                        "Original": [input_text],
                        "Readability_Level": [i],
                        "Simplification": [response],
                        "Predicted_Level": [prediction],
                    }
                )

                df = pd.concat([df, new_row], ignore_index=True)
                pbar.update()

        # Write DataFrame to an Excel file
        df.to_excel(f"Simplifications_GEITje({start}-{end}).xlsx", index=False)

        start += 5
        end += 5

    pbar.close()

        

if __name__ == "__main__":
    main()
