import pandas as pd


def create_nusselt_dataframe(columns, **metadata):
    """
    Create a DataFrame with two rows (Nu_hot, Nu_cold) and the specified columns.
    Optionally attach metadata via keyword arguments.

    Parameters:
    - columns (list of str): Column names (e.g., turbulence models)
    - metadata (kwargs): Optional metadata to attach via df.attrs

    Returns:
    - pd.DataFrame: Initialized DataFrame with metadata
    """
    df = pd.DataFrame(
        {col: [None, None] for col in columns},
        index=["Nu_hot", "Nu_cold"],
    )
    df.index.name = "Quantity"
    df.attrs.update(metadata)
    return df


def convert_labels(label_list):
    """Convert LaTeX-formatted labels to plain text by removing special characters."""
    def rm(string, char):
        return string.replace(char, "")

    converted = []
    for label in label_list:
        new_label = label.strip()
        for char in ("$", "^", "\\"):
            new_label = rm(new_label, char)
        converted.append(new_label)
    return converted
