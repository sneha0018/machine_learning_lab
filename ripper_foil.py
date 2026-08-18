# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================

# Run this in the VS Code terminal:
# pip install wittgenstein pandas scikit-learn


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
from sklearn.datasets import load_iris
import wittgenstein as lw


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Load the Iris dataset
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

# Combine features and target into one DataFrame
df = pd.concat([X, y.rename("target")], axis=1)

print("Dataset Head:")
print(df.head())


# ============================================================
# STEP 2: RIPPER ALGORITHM
# ============================================================

# Create RIPPER model
model = lw.RIPPER()

# Specify positive class
# Here, class 0 represents Setosa
model.fit(X, y, pos_class=0)

print("\n=== RIPPER Rules ===")
print(model.ruleset_)


# ============================================================
# STEP 3: PREPARE DATA FOR FOIL
# ============================================================

# Create a binary classification dataset
# 1 = Setosa
# 0 = Not Setosa

df_bin = df.copy()

df_bin["target"] = (df_bin["target"] == 0).astype(int)


# List of attributes
attributes = list(X.columns)


# ============================================================
# STEP 4: FOIL INFORMATION GAIN
# ============================================================

def foil_gain(pos_before, neg_before, pos_after, neg_after):

    # If there are no positive examples after
    # applying the condition, return a very small value
    if pos_after == 0:
        return -1e9

    # FOIL information gain formula
    return pos_after * (
        (pos_after / (pos_after + neg_after))
        - (pos_before / (pos_before + neg_before))
    )


# ============================================================
# STEP 5: FOIL ALGORITHM
# ============================================================

def foil(df, target_col="target"):

    rules = []

    # Count positive and negative examples
    pos_total = df[target_col].sum()
    neg_total = len(df) - pos_total

    # Continue until all positive examples are covered
    while pos_total > 0:

        rule = []

        pos_rem = pos_total
        neg_rem = neg_total

        # Initially, all examples are covered
        covered = df.copy()

        # Continue adding conditions until no negatives remain
        while neg_rem > 0:

            best_gain = -1e9
            best_attr = None
            best_val = None
            best_subset = None

            # Try every attribute
            for attr in attributes:

                # Try every possible value
                for val in df[attr].unique():

                    # Select examples satisfying the condition
                    subset = covered[covered[attr] == val]

                    pos_after = subset[target_col].sum()
                    neg_after = len(subset) - pos_after

                    # Calculate FOIL gain
                    gain = foil_gain(
                        pos_rem,
                        neg_rem,
                        pos_after,
                        neg_after
                    )

                    # Keep the condition with the highest gain
                    if gain > best_gain:
                        best_gain = gain
                        best_attr = attr
                        best_val = val
                        best_subset = subset

            # Stop if no useful condition is found
            if best_attr is None:
                break

            # Add the best condition to the rule
            rule.append((best_attr, best_val))

            # Update covered examples
            covered = best_subset

            # Update positive and negative counts
            pos_rem = covered[target_col].sum()
            neg_rem = len(covered) - pos_rem

        # Add the learned rule
        rules.append((rule, 1))

        # Remove covered examples
        df = df.drop(covered.index)

        # Recalculate positive and negative examples
        pos_total = df[target_col].sum()
        neg_total = len(df) - pos_total

    return rules


# ============================================================
# STEP 6: RUN FOIL
# ============================================================

rules = foil(df_bin)


# ============================================================
# STEP 7: DISPLAY FOIL RULES
# ============================================================

print("\n=== FOIL Learned Rules (Setosa vs Not) ===")

for conditions, prediction in rules:

    condition_string = " AND ".join(
        [f"{attribute}={value}" for attribute, value in conditions]
    )

    print(
        f"IF {condition_string} "
        f"THEN class={prediction}"
    )