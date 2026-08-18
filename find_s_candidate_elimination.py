import pandas as pd


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Example dataset: EnjoySport
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=[
    'Sky',
    'AirTemp',
    'Humidity',
    'Wind',
    'Water',
    'Forecast',
    'EnjoySport'
])

print("Dataset:")
print(data)


# Attributes and target
attributes = data.columns[:-1]
target = data.columns[-1]


# ============================================================
# STEP 2: FIND-S ALGORITHM
# ============================================================

def find_s(df):
    # Start with the most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 1)

    # Check each training example
    for i, row in df.iterrows():

        # Consider only positive examples
        if row[target] == "Yes":

            for j in range(len(hypothesis)):

                # If hypothesis is empty, assign the value
                if hypothesis[j] == '0':
                    hypothesis[j] = row.iloc[j]

                # If values are different, generalize to '?'
                elif hypothesis[j] != row.iloc[j]:
                    hypothesis[j] = '?'

    return hypothesis


# Run Find-S
final_hypothesis = find_s(data)

print("\nMost specific hypothesis (Find-S):")
print(final_hypothesis)


# ============================================================
# STEP 3: CANDIDATE ELIMINATION ALGORITHM
# ============================================================

def candidate_elimination(df):

    # Initialize S (Most Specific Boundary)
    S = [['0'] * (len(df.columns) - 1)]

    # Initialize G (Most General Boundary)
    G = [['?'] * (len(df.columns) - 1)]

    # Process every training example
    for i, row in df.iterrows():

        # ----------------------------------------------------
        # POSITIVE EXAMPLE
        # ----------------------------------------------------
        if row[target] == "Yes":

            # Remove hypotheses from G that are inconsistent
            # with the positive example
            G = [
                g for g in G
                if all(
                    g[j] == '?' or g[j] == row.iloc[j]
                    for j in range(len(attributes))
                )
            ]

            # Update S
            for j in range(len(attributes)):

                if S[0][j] == '0':
                    S[0][j] = row.iloc[j]

                elif S[0][j] != row.iloc[j]:
                    S[0][j] = '?'

        # ----------------------------------------------------
        # NEGATIVE EXAMPLE
        # ----------------------------------------------------
        else:

            # Remove hypotheses from S that are inconsistent
            # with the negative example
            S = [
                s for s in S
                if not all(
                    s[j] == '?' or s[j] == row.iloc[j]
                    for j in range(len(attributes))
                )
            ]

            # Specialize G
            new_G = []

            for g in G:

                for j in range(len(attributes)):

                    if g[j] == '?':

                        # Get all possible values of this attribute
                        for val in df[attributes[j]].unique():

                            # Value should differ from negative example
                            if val != row.iloc[j]:

                                new_hypo = g.copy()
                                new_hypo[j] = val

                                # Check consistency with S
                                if any(
                                    all(
                                        s[k] == '?'
                                        or s[k] == new_hypo[k]
                                        or s[k] == '0'
                                        for k in range(len(attributes))
                                    )
                                    for s in S
                                ):
                                    new_G.append(new_hypo)

            G = new_G

    return S, G


# Run Candidate Elimination
S_final, G_final = candidate_elimination(data)


# ============================================================
# STEP 4: DISPLAY RESULTS
# ============================================================

print("\nCandidate Elimination Results:")

print("S (Most Specific Boundary):")
print(S_final)

print("\nG (Most General Boundary):")
print(G_final)