# End-to-End Sentiment Analysis Pipeline & RAG Chatbot

This repository contains two projects:

# 1) End-to-End Sentiment Analysis Pipeline :
      A machine learning pipeline to classify IMDB movie reviews as positive or negative using a Logistic Regression or Naive Bayes model.

# 2)RAG (Retrieval-Augmented Generation) Chatbot :
      A chatbot system that uses vector databases to fetch relevant answers based on user queries and generates responses with a pre-trained generative model.

# Project 1: End-to-End Sentiment Analysis Pipeline

  
# Overview

        The End-to-End Sentiment Analysis Pipeline classifies IMDB movie reviews into positive or negative sentiment using a machine learning model. 
        It includes steps for data collection, preprocessing, training, evaluation, and serving the model through an API.

# Features


1)Data Collection: Utilizes the IMDB dataset for movie reviews.

2)Model: Sentiment analysis model using Logistic Regression or Naive Bayes.

3)API: Flask-based API for real-time sentiment prediction.

4)Model Evaluation: Evaluation metrics such as accuracy, precision, recall, and F1-score.

# Requirements:
    Python 3.x

# Libraries:
    1)Flask
    
    2)scikit-learn
    
    3)pandas
    
    4)numpy
    
    5)SQLite3



# Install dependencies:
    
    pip install -r requirements.txt

# Set up the Database:

        SQLite: sqlite3 imdb_reviews.db < data_setup.sql

# Steps to Run
          
          Data Setup: Download the IMDB dataset from Kaggle and place it in the data/ folder .


# python
      from datasets import load_dataset

      dataset = load_dataset("imdb")


# Run data_setup.py to load data into the database:

            python data_setup.py


# Model Training: Train the sentiment analysis model:

            python train_model.py


# Start the Flask API:

        python app.py

# Test the API: Send a POST request to the /predict endpoint:


        curl -X POST -H "Content-Type: application/json" -d '{"review_text": "This movie was amazing!"}' http://localhost:5000/predict

# Model Evaluation

The evaluation results (accuracy, precision, recall, and F1-score) will be displayed after the model training.

# Project Structure
       
        1)data_setup.py: Prepares the dataset and loads it into the database.

        2)train_model.py: Trains the sentiment analysis model.
        
        3)app.py: Flask API for real-time prediction.

        4)data_setup.sql: SQL commands for database setup.
        
        5)requirements.txt: Python dependencies.



# Project 2: RAG (Retrieval-Augmented Generation) Chatbot:

# Overview


The RAG chatbot system uses vector databases to store text data, retrieves the most relevant documents for a given query, and generates human-like responses using a generative model.

# Features



        1)Text Corpus: Preprocesses and stores a text corpus (e.g., Wikipedia, documentation) for retrieval.
        
        2)Vector Store: Stores document embeddings using Faiss/Chroma/Milvus.
        
        3)RAG Model: Combines document retrieval with a generative model to create context-based responses.
        
        4)API: A Flask API for chatbot interactions.


# Requirements:


      Python 3.x

# Libraries:


      1)Flask
      
      2)sentence-transformers
  
      3)Faiss/Chroma/Milvus
      
      4)numpy
      
      5)SQLite3

# Software:

    SQLite3 for storing chat history

# Install dependencies:


        pip install -r requirements.txt


# Set up the SQLite3 Database Create tables for storing chat history:


          mysql -u root -p < create_tables.sql

# Steps to Run:


Prepare the Data: Preprocess the text corpus (e.g., Wikipedia or documentation) and clean it for embedding.


# Run prepare_data.py:


    python prepare_data.py


# Vector Database Setup: Embed the text data and store it in Faiss/Chroma/Milvus:


            python vector_db_setup.py


# Start the Flask API:

       python app.py


# Test the Chatbot: Send a POST request to the /chat endpoint with the user query:


 curl -X POST -H "Content-Type: application/json" -d '{"query": "What is AI?"}' http://localhost:5000/chat


# View Chat History: You can view chat history by sending a GET request to /history:


               curl- http://localhost:5000/history


# Chatbot Architecture:

            1)Embedding Generation: Uses the sentence-transformers library to create embeddings for each document chunk.

            2)Query Handling: When a user asks a question, the query is embedded, and the top-k relevant documents are retrieved from the vector store.
            
            3)Response Generation: The retrieved documents are used as context to generate a response using a pre-trained generative model.


# Project Structure:

            1)prepare_data.py: Preprocesses and cleans the dataset.
            
            2)vector_db_setup.py: Embeds data and stores it in the vector database.
            
            3)app.py: Flask app that serves the chatbot via API.
            
            4)create_tables.sql: SQL script for creating chat history tables.
            
            5)requirements.txt: Python dependencies for the project.


# Testing:

# Unit tests for embedding generation and retrieval can be found in test_embedding.py. To run tests:

              pytest test_embedding.py

