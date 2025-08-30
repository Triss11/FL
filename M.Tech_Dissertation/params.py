# --- Inputs for dataset splitting ---

# "binary" or "multiclass"  (case-insensitive)
classification = "binary"

# "fnn" or "lstm"  (case-insensitive)
model = "lstm"

# number of users (1..128)
users = 128

# reproducibility
seed = 42

# Path to the single CSV with both features and labels
DATASET_CSV = "/Users/sohinikar/FL/M.Tech_Dissertation/Obfuscated-MalMem2022.csv"

# Name of the label column; if None, auto-detects from common names:
# ["label","Label","class","Class","target","Target","y","Y"]
label_column = None

# Where to save the CSV that excludes the 5,888 sampled rows.
# If None, the script will create "<original_stem>_unseen.csv" in the same folder.
UNSEEN_OUT_CSV = None

# If True, remove any existing user_* folders under the chosen destination before writing
clean_existing = True