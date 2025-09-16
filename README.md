# BMI Prediction Using Metagenomic Data

This project focuses on predicting Body Mass Index (BMI) using microbiome data through various linear machine learning regression models.

## Project Structure

## Overview

The project implements machine learning models to predict BMI values using metagenomic data. It includes data preprocessing, model training, evaluation, and overfitting analysis.

### Key Components

- **Data Processing**: Cleaning and preprocessing of metagenomic data
- **Model Training**: Implementation of various regression models
- **Evaluation**: Performance assessment using metrics like R² and RMSE
- **Overfitting Analysis**: Detection and visualization of model overfitting

## Getting Started

### Installation

1. Clone the repository:
git clone [repository-url]
cd Assignment-1

2. Install required packages:
micromamba env create -f environment.yml

## Usage

1. Data Preprocessing:
   - Located in `data/` directory

2. Model Training:
   - Navigate to `notebooks/model_analysis.ipynb`
   - Runs multiple regression models on the preprocessed data
   - Evaluates model performance on validation set

3. Analysis:
   - Feature importance visualization
   - Model performance comparison
   - Overfitting detection and analysis
