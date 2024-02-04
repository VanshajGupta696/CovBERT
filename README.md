# CovBERT Project

## Overview
CovBERT is a comprehensive analysis and machine learning project focused on COVID-19 tweets. It utilizes various data preprocessing techniques, exploratory data analysis (EDA), and implements machine learning and deep learning models to understand sentiments and other patterns in COVID-19 related tweets.

## Features
- Data cleaning and preprocessing.
- EDA of geolocation and sentiment data from tweets.
- Sentiment analysis using traditional machine learning models (Naive Bayes, SVM, Logistic Regression) and deep learning models (LSTM, BERT).
- Geolocation analysis to understand the distribution of tweets across different regions.
- Custom text cleaning function to prepare text data for model training.

## Installation
To run the CovBERT project, you need to have Python installed on your machine. It's recommended to use a virtual environment. Follow these steps to set up the project environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/CovBERT.git
cd CovBERT

# (Optional) Setup a virtual environment
python -m venv covbert-env
source covbert-env/bin/activate  # On Windows, use `covbert-env\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```


## Usage

This project is structured to analyze COVID-19 related tweets using both machine learning and deep learning techniques. Follow the steps below to get started with CovBERT:

### Data Preparation

1. Ensure your datasets are located in the appropriate directory. By default, the project expects data files to be placed in a directory named `data/` at the root of the project. If your data is stored elsewhere, update the file paths in the scripts accordingly.

2. The main datasets used in this project are:
   - `covid19_tweets.csv`: Contains tweets related to COVID-19, including the tweet text and metadata.
   - `finalSentimentdata2.csv`: Contains processed tweets with sentiment labels.

### Running the Analysis

To perform the data analysis and execute the machine learning models:

1. Open the project in your preferred development environment. If you are using Jupyter Notebooks, navigate to the notebook file (e.g., `CovBERT.ipynb`).

2. Run the notebook cells in sequence to preprocess the data, explore the data through visualizations, and train the models.

### Models

The project implements several models for sentiment analysis and geolocation analysis:

- **Sentiment Analysis**: Utilizes Naive Bayes, SVM, Logistic Regression, LSTM, and BERT models to classify the sentiment of tweets into categories such as positive, negative, and neutral.

- **Geolocation Analysis**: Analyzes the distribution of tweets by geolocation data to understand how different regions are discussing COVID-19.

### Visualization

Various visualizations are included to aid in the analysis:

- **Word Clouds**: For understanding the most common words in the dataset.
- **Bar Charts and Pie Charts**: For distribution of sentiments and tweet counts by region.
- **Choropleth Maps**: For a geographical distribution of tweets.

### Custom Functions

Custom utility functions are provided for data cleaning, preprocessing, and model evaluation to streamline the analysis process.

## Contributing

Contributions to the CovBERT project are welcome. Please follow the standard GitHub pull request process to submit your contributions. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Project Owner**: Vanshaj Gupta
- **Email**: vanshajgupta617@gmail.com

## Acknowledgments

This project wouldn't have been possible without the help of the following tools and libraries:

- [Google Colab](https://colab.research.google.com/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

A huge thanks to all contributors and open-source libraries that made this project possible.
