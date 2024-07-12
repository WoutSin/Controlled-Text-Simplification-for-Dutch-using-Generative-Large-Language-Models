import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_excel_and_label(path):
    """
    Reads an Excel file and labels the data based on readability levels.

    Args:
        path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The combined DataFrame with readability levels.
    """
    xl = pd.ExcelFile(path)
    sheet_names = xl.sheet_names
    dfs = []
    for i, sheet in enumerate(sheet_names):
        df = xl.parse(sheet)
        df["readability_level"] = i + 1
        dfs.append(df)
    df_combined = pd.concat(dfs)
    return df_combined

def plot_histograms(df, features, output_dir):
    """
    Plots and saves histograms for the specified features, split by readability levels.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list): List of feature names to plot histograms for.
        output_dir (str): The directory where the images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    readability_levels = df["readability_level"].unique()
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        for level in readability_levels:
            level_data = df[df["readability_level"] == level][feature]
            n, bins, patches = plt.hist(level_data, bins=30, alpha=0.5, label=f'Readability Level {level}')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plt.plot(bin_centers, n, '-o', label=f'Level {level}')
        
        plt.title(f'Histogram and Line Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_histogram_line.png'))
        plt.close()

def main():
    # Path to the Excel file
    path = 'Corpus_Selection.xlsx'  # Update this with the actual path
    
    # Output directory for images
    output_dir = 'histogram_images'
    
    # Read and label the data
    df = read_excel_and_label(path)
    
    # List of features to plot histograms for
    features = [
        "Let_per_wrd",
        "Morf_per_wrd",
        "Freq1000",
        "Freq2000",
        "Freq3000",
        "Freq5000",
        "Freq10000",
        "Freq20000",
        "Namen_d",
        "MTLD_inhwrd",
        "MTLD_inhwrd_zonder_abw",
        "Pers_nw_d",
        "Plaats_nw_d",
        "Organisatie_nw_d",
        "Pers_vnw_d",
        "Org_namen_d",
        "Spec_d",
    ]
    
    # Plot the histograms
    plot_histograms(df, features, output_dir)

if __name__ == "__main__":
    main()
