import pandas as pd


# Constructs a balanced training dataset across a given 'train_concept'
# (e.g., gender) while controlling for a second 'independent_concept' (e.g., genre) to avoid confounding.
def get_train_dataset(metadata, train_concept, independent_concept, max_test_size=100):
    df_train = {}
    for train_class in train_concept[1]:  # e.g., 'female', 'male'
        df_train[train_class] = []
        for independent_class in independent_concept[1]:  # e.g., genres
            # Filter by independent attribute (e.g., genre = Rock)
            df_sub = metadata[metadata[independent_concept[0]] == independent_class]
            df_sub_positive = df_sub[df_sub[train_concept[0]] == train_class]
            df_sub_negative = df_sub[~(df_sub[train_concept[0]] == train_class)]

            # Balance the number of positive and negative samples
            minimum_counts = min(len(df_sub_positive), len(df_sub_negative))
            if minimum_counts > max_test_size:
                df_sub_positive = df_sub_positive.sample(minimum_counts - max_test_size)
                df_sub_negative = df_sub_negative.sample(minimum_counts - max_test_size)
                df_sub_positive["target"] = 1
                df_sub_negative["target"] = 0
                df_train[train_class].append(df_sub_positive)
                df_train[train_class].append(df_sub_negative)

        # Combine all genre-balanced samples
        df_train[train_class] = pd.concat(df_train[train_class])
    return df_train


# Constructs a balanced test set with no overlap with the training data.
def get_test_dataset(metadata, train_dataset, train_concept, independent_concept):
    df_test = {}
    for train_class in train_concept[1]:
        df_test[train_class] = []
        # Exclude training samples
        df_not_train = metadata[
            ~metadata.song_id.isin(train_dataset[train_class].song_id)
        ]

        for independent_class in independent_concept[1]:
            df_sub = df_not_train[
                df_not_train[independent_concept[0]] == independent_class
            ]
            df_sub_positive = df_sub[df_sub[train_concept[0]] == train_class]
            df_sub_negative = df_sub[~(df_sub[train_concept[0]] == train_class)]

            # Balance positives and negatives
            minimum_counts = min(len(df_sub_positive), len(df_sub_negative))
            df_sub_positive = df_sub_positive.sample(minimum_counts)
            df_sub_negative = df_sub_negative.sample(minimum_counts)
            df_sub_positive["target"] = 1
            df_sub_negative["target"] = 0

            df_test[train_class].append(df_sub_positive)
            df_test[train_class].append(df_sub_negative)

        df_test[train_class] = pd.concat(df_test[train_class])
    return df_test


# High-level wrapper for generating training and test sets, stratified by genre.
def get_genre_balanced_datasets(metadata, train_concept):
    # Drop missing labels for the training concept
    song_info_without_nan = metadata[metadata[train_concept[0]].notnull()]
    independent_concept = ("genre", list(metadata.genre.unique()))

    train_dataset = get_train_dataset(
        metadata=song_info_without_nan,
        train_concept=train_concept,
        independent_concept=independent_concept,
    )

    test_dataset = get_test_dataset(
        metadata=song_info_without_nan,
        train_dataset=train_dataset,
        train_concept=train_concept,
        independent_concept=independent_concept,
    )

    return train_dataset, test_dataset


def preprocess_metadata(song_info_path, song_artist_path, artist_info_path):
    """Utility function to load and merge raw metadata CSVs."""
    metadata = (
        pd.read_csv(song_info_path)
        .merge(pd.read_csv(song_artist_path))
        .merge(pd.read_csv(artist_info_path))
    )
    metadata = metadata.drop_duplicates(keep="first").reset_index(drop=True)

    return metadata
