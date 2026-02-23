import pandas as pd
from sklearn.utils import resample

INPUT_CSV = "dataset/HAM10000_metadata.csv"
OUTPUT_CSV = "dataset/balanced_metadata.csv"
SAMPLES_PER_CLASS = 300
RANDOM_STATE = 42

df = pd.read_csv(INPUT_CSV)

balanced_dfs = []

for label in df['dx'].unique():
    class_df = df[df['dx'] == label]

    if len(class_df) >= SAMPLES_PER_CLASS:
        # Undersample large classes
        sampled_df = resample(
            class_df,
            replace=False,
            n_samples=SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )
    else:
        # Oversample small classes
        sampled_df = resample(
            class_df,
            replace=True,   # 🔑 KEY CHANGE
            n_samples=SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )

    balanced_dfs.append(sampled_df)

balanced_df = pd.concat(balanced_dfs)

balanced_df.to_csv(OUTPUT_CSV, index=False)

print("Balanced dataset created with ALL 7 classes ✅")
print(balanced_df['dx'].value_counts())