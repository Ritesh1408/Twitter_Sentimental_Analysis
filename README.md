# Twitter_data
# Sentimental Analysis Using ML

Code Link and Output :- https://colab.research.google.com/drive/1sv7YKW62XG-NLvU8-LrQUeRGMnoBzx3Q?usp=sharing
Dataset Link : https://www.kaggle.com/datasets/kazanova/sentiment140


Tools used :-  Google Collab
Language : Python
Machine Learning Model.


# Twitter Sentiment Analysis using Machine Learning

This project focuses on sentiment analysis of tweets using machine learning techniques. 
It utilizes a dataset from Twitter to train a model that can classify tweets as positive or negative based on their content.

## Steps to Run the Project

1. **Setup Kaggle API:**
   - Install the Kaggle library using pip: `!pip install kaggle`.
   - Upload your Kaggle API key (`kaggle.json`) file.
   - Configure the path of the `kaggle.json` file.

2. **Download Dataset:**
   - Use the Kaggle API to download the sentiment dataset: `!kaggle datasets download -d kazanova/sentiment140`.
   - Extract the dataset from the compressed file.

3. **Data Processing:**
   - Import necessary libraries and dependencies.
   - Load the dataset into a pandas DataFrame.
   - Check for missing values and data distribution.
   - Convert the target labels (`4` to `1` for positive sentiment).
   - Perform text preprocessing including stemming to reduce words to their root form.

4. **Splitting Data:**
   - Split the dataset into training and testing data using `train_test_split`.

5. **Feature Extraction:**
   - Convert textual data into numerical data using TF-IDF vectorization.

6. **Training the Model:**
   - Utilize Logistic Regression for sentiment classification.
   - Train the model on the training data.

7. **Model Evaluation:**
   - Evaluate the model's accuracy on both training and testing data.

8. **Saving the Model:**
   - Save the trained model using pickle for future use.

9. **Predictions:**
   - Use the saved model to make predictions on new data (tweets).

## Model Details

- **Algorithm Used:** Logistic Regression
- **Accuracy on Training Data:** 77.8%
- **Framework/Libraries Used:** Python (NumPy, pandas, scikit-learn, NLTK)

## File Structure

- `Twitter Sentiment Analysis using ML.ipynb`: Google Collab Notebook containing the project code.
- `trained_model.sav`: Saved trained model for future predictions.


