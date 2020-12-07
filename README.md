newsentity 📰
============

[![](https://img.shields.io/badge/Made_with-Python3-red?style=for-the-badge&logo=python)]()
[![](https://img.shields.io/badge/Made_with-nltk-red?style=for-the-badge&logo=nltk)]()
[![](https://img.shields.io/badge/Made_with-pandas-red?style=for-the-badge&logo=pandas)]()
[![](https://img.shields.io/badge/Made_with-pillow-red?style=for-the-badge&logo=pillow)]()
[![](https://img.shields.io/badge/Made_with-afinn-red?style=for-the-badge&logo=afinn)]()
[![](https://img.shields.io/badge/Made_with-plotly-red?style=for-the-badge&logo=plotly)]()
[![](https://img.shields.io/badge/Made_with-streamlit-red?style=for-the-badge&logo=streamlit)]()
[![](https://img.shields.io/badge/deployed_on-heroku-red?style=for-the-badge&logo=heroku)]()

A dashboard for analysing entities (people and organisations) in the content of recent news articles + media coverage by user sentiment.

It uses the [LatestNews](https://algorithmia.com/algorithms/specrom/LatestNewsAPI) API for getting news content. You can find the deployed website [here!](https://newsentity.herokuapp.com/) 

---

## Features:

- Named entities (person or organisation) can be filtered by number to obtain graphs and wordclouds.
- Selecting a particular entity finds other trending topics in the news related to it + groups the various news sources covering the topic to find the intensity of their sentiments using the Afinn Lexicon.
- The `sentiment.py` file creates the dashboard by loading the relevant files. It uses NLTK'S Parts of Speech tagger to chunk the tokens based on their POS tags to find named entities.
- The `ner.ipynb` file contains previous attempts to solve the problem by using scikit-learn's CountVectorizer.
- The dashboard currently only works on a small subset of data for the purpose of this experiment. Expanding it to cover news articles daily remains in the future scope.

---

## Screenshots:

<p float="left">
  <img src="https://user-images.githubusercontent.com/50259869/101362865-afdb5880-38c6-11eb-8972-cd23ef385313.png" width="300" />
  <img src="https://user-images.githubusercontent.com/50259869/101362951-c7b2dc80-38c6-11eb-8320-5595d818dbd7.png" width="300" /> 
</p>
<p float="left">
  <img src="https://user-images.githubusercontent.com/50259869/101364116-3a708780-38c8-11eb-82a8-0cb9f56028a6.png" width="300" />
  <img src="https://user-images.githubusercontent.com/50259869/101364184-4eb48480-38c8-11eb-8030-2688e7e2e075.png" width="300" /> 
</p>

---

## Tech stack:

- `nltk:` POS tagging to perform NER by chunking tokens.
- `pandas:` formatting and cleaning the data.
- `afinn:` lexicon to measure coverage sentiment.
- `plotly express:` visualisations.
- `streamlit:` web framework.

---

## Deployment:

The live project is deployed on https://newsentity.herokuapp.com/. 

---

## Local installation:

**You must have Python 3.6 or higher to run the file.**

- Create a new virtual environment for running the application. You can follow the instructions [here.](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/)
- Navigate to the virtual environment and activate it.
- Install the dependancies using `pip install -r requirements.txt`
- Run the `news.py` file with `streamlit run news.py`