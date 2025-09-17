# Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors - Official Implementation

[![Paper](https://img.shields.io/badge/paper-ISMIR:<id>-024291)](./paper_ismir_2025.pdf)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This repository provides the code implementation for the ISMIR2025 publication [Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors](./paper_ismir_2025.pdf).

## Overview

This repository consists of code an Jupyter notebooks used of preparation of audio embeddings, training of classifiers for CAV vector extraction and evaluation of those CAV vectors.

```
/
├── data
│   └── create_datasets.ipynb                 # Notebook for dataset preparation
├── embedding_calculation
│       └── calculate_mert_embeddings.ipynb   # Notebooks for audio embedding calculation
|       └── ...
├── evaluation
│   └── evaluate_tcav.ipynb                   # Notebook for evaluating trained CAV vectors
├── packages
│   └── cavmir                                # python package containing code used for
|       └── ...                               # embedding calcualtion and CAV training
├── train_scripts
│   └── train_cavs.ipynb                      # Notebook for training CAV vectors

```

## Setup

This repoitory requires Python 3.11 and a Jupyter Notebook compatible IDE

Setup a new virtual environment, clone the repository and install the provided python package:

```bash
git clone git@github.com:WhoCares96/CAV-MIR.git
cd CAV-MIR

python -m venv .venv
source .venv/bin/activate

pip install [-e] packages/cavmir
```

## Dataset

The dataset used for training the Concept Activation Vectors in this paper is "STraDa: A Singer Traits Dataset". The associated metadata files as well as explanations on how to access the audio files can be found on [Zenodo](https://zenodo.org/records/10057434).

As described in the paper, in addition to the STraDa data, we manually annotated extra songs to even out underrepresented categories. The corresponding metadata can be found [here](./data/supplementary_dataset.txt)

After retrieval of the metadata files, please run the corresponding [create_datasets.ipynb](./data/create_datasets.ipynb) notebook to prepare genre-balanced datasets for later use.

## Usage

In order to reproduce the paper results please do the following:

#### 1. Prepare dataset

Follow the instructions in the [Dataset](#dataset) section to retrieve the dataset metadata and the corresponding audio files. The provided notebooks expect data to be stored on AWS S3 but the package uses a fsspec abstraction, thus can be easily adatpted for other storage types. Make sure that all audio files are stored in a single location in the format `<deezer-id>.mp3`.

#### 2. Provide .env file

The Jupyter Notebooks make use of dotenv to access the locations of the dataset. Duplicate the [.env.sample](.env.sample) and provide the following:

- `AUDIO_PREFIX`: path or uri to the stored audio files
- `DATASET_PREFIX`: path or uri to the dataset files created prior
- `EMBEDDING_PREFIX`: path or uri to store audio embeddings to

#### 3. Calculate audio embeddings

Head to [embedding_calculation](./embedding_calculation) and run all notebooks to obtain embeddings from the audio files.

#### 4. Run CAV vector trainings

Open the [train_cavs.ipynb Notebook](./train_scripts/train_cavs.ipynb) and run it to create CAV vectors from the embeddings.

#### 5. Run evaluation

Open the [evaluate_tcav.ipynb Notebook](./evaluation/evaluate_tcav.ipynb) and run it to obtain plots for all trained genre-concept combinations.

## Citation

```text
@inproceedings{gebhardt2025cav,
  title     = {Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors},
  author    = {Roman B. Gebhardt and Arne Kuhle and Eylül Bektur},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2025}
}
```

## License

This repository is released under the MIT license. Please see the [LICENSE](LICENSE) file for more details.

## Contact

For any enquiries regarding this repository and the scripts usage, please open an issue or contact us via [cavmir@cyanite.ai](mailto:cavmir@cyanite.ai)
