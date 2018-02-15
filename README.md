The project has been developed using python3.


Required packages:

    sklearn
    scikit-learn
    scipy
    pandas
    pandas-datareader
    matplotlib
    numpy
    

How to run:
    
    Simply execute all commands in "main.py".
    The code for testing each feature (at the bottom of the file) has been commented out
    because it takes a very long time to execute (i.e. several hours).
    I selected the best features for the last test based on the output of those long tests.

    
Code description:
    
    data_gather.py
        Reads data from files and performs basic data preprocessing.
    feature_generation.py
        Performs most data processing operations and generates new features from existing data.
    learning.py
        Mostly deals with applying machine learning algorithms.
        It also contains a few functions for preparing the dataset for training/testing. 
    main.py
        Performs the actual tests. 
    util.py
        Utility functions.

        
Datasets:
    
    Datasets are freely available at Yahoo Finance but, since I slightly modified them
    (i.e. removed some null values), I decided to upload them too.
