# Importing all needed libraries
## Machine learing and optimization
import os
import joblib
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from skbio.stats.composition import clr, multi_replace
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression, VarianceThreshold
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
## Handling warnings
import warnings
warnings.filterwarnings("ignore")


# ----------------- Helper Functions for data processing ----------------- #
COLUMNS_TO_DROP = ['Unnamed: 0', 'Project ID', 'Experiment type', 'Disease MESH ID', 'Sex']

class EmptySampleFilter(BaseEstimator, TransformerMixin):
    """Custom transformer to filter samples with all zeros or only one non-zero feature in features."""
    def __init__(self, target_col=None):
        self.target_col = target_col
        self.valid_samples = None
    
    def fit(self, X, y=None):
        # Check which samples have all zeros or only one non-zero feature in features (excluding target if in X)
        if isinstance(X, pd.DataFrame):
            if self.target_col and self.target_col in X.columns:
                features = X.drop(columns=[self.target_col])
            else:
                features = X
            self.valid_samples = (features != 0).sum(axis=1) > 1
        else:  # numpy array
            self.valid_samples = (X != 0).sum(axis=1) > 1
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.valid_samples]
        else:  # numpy array
            return X[self.valid_samples, :]

class CompositionalTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to handle zero-inflated compositional data."""
    def __init__(self, exclude_col=None):
        self.exclude_col = exclude_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.exclude_col and self.exclude_col in X.columns:
            excluded_data = X[self.exclude_col]
            X = X.drop(columns=[self.exclude_col])
            X_imputed = (X + 1e-5)
            transformed = clr(X_imputed)
            transformed[self.exclude_col] = excluded_data.values
            return transformed
        else:
            X_imputed = multi_replace(X)
            return clr(X_imputed)

def data_clean(dataset_path: pd.DataFrame, cols_drop: list = COLUMNS_TO_DROP) -> pd.DataFrame:
    """
    Cleans the dataset by dropping user specified columns and checks for missing values.
    Args:
        dataset_path (str): Path to the dataset.
        cols_drop (list): List of columns to drop from the dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    raw_data = pd.read_csv(dataset_path)
    raw_data.drop(columns=cols_drop, inplace=True)
    if raw_data.isna().any().any():
        print("\nWarning: NA values detected!")
        print("Rows with NAs:")
        print(raw_data[raw_data.isna().any(axis=1)].head())
    else:
        print("\nNo NA values found.")
    
    return raw_data
    
def analyze_feature_importance(pipeline, feature_names):
    """
    Enhanced feature importance analysis that supports feature selection and models without coefficients.
    
    Args:
        pipeline: The trained pipeline containing the model and optional feature selection step.
        feature_names: List of feature names before feature selection.
    
    Returns:
        pd.DataFrame: DataFrame containing feature names and their importance scores.
    """
    try:
        # Handle feature selection if present
        if 'feature_selection' in pipeline.named_steps:
            selected = pipeline.named_steps['feature_selection'].get_support()
            feature_names = feature_names[selected]

        # For models with coef_ (e.g., linear models)
        if hasattr(pipeline.named_steps['model'], 'coef_'):
            coef = pipeline.named_steps['model'].coef_
            if len(coef.shape) > 1:  # For multi-output models
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)

        # For models without coefficients or feature importances (e.g., SVR)
        else:
            print(f"Warning: Model {type(pipeline.named_steps['model']).__name__} does not support feature importance.")
            return pd.DataFrame({'feature': feature_names, 'importance': [None] * len(feature_names)})

        # Return feature importance as a DataFrame
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return pd.DataFrame()


# ----------------- Helper Functions for machine learning ----------------- #


def optimize_optuna(X, y, pipeline, param_space, n_trials=100, cv=5, scoring='neg_root_mean_squared_error', verbose_level=1, random_state=42):
    """
    Bayesian hyperparameter optimization using Optuna.
    
    Args:
        X: Features
        y: Target
        pipeline: sklearn Pipeline or estimator
        param_space: Dictionary of parameter distributions
        n_trials: Number of optimization trials
        cv: Cross-validation strategy
        scoring: Scoring metric
        random_state: Random seed
        
    Returns:
        best_params: Dictionary of optimized parameters
        best_score: Best CV score achieved
        study: Optuna study object for further analysis
    """
    def objective(trial):
        params = {}
        for param_name, param_dist in param_space.items():
            if isinstance(param_dist, tuple):  # Correctly formatted distribution
                if param_dist[0] == 'log-uniform':
                    params[param_name] = trial.suggest_float(param_name, param_dist[1], param_dist[2], log=True)
                elif param_dist[0] == 'uniform':
                    params[param_name] = trial.suggest_float(param_name, param_dist[1], param_dist[2])
                elif param_dist[0] == 'int-uniform':
                    params[param_name] = trial.suggest_int(param_name, param_dist[1], param_dist[2])
                else:
                    raise ValueError(f"Unsupported distribution type: {param_dist[0]}")
            elif isinstance(param_dist, list):  # Categorical
                params[param_name] = trial.suggest_categorical(param_name, param_dist)
            else:
                raise ValueError(f"Invalid parameter distribution format for {param_name}: {param_dist}")
        
        # Clone and set parameters
        current_pipeline = clone(pipeline)
        current_pipeline.set_params(**params)
        
        # Calculate cross-validation score
        scores = cross_val_score(
            current_pipeline, X, y, cv=cv, 
            scoring=scoring, n_jobs=-1
        )
        return np.mean(scores)
    
    # Create and run study
    optuna.logging.set_verbosity(optuna.logging.WARNING if verbose_level < 2 else optuna.logging.INFO)
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    
    # Control Optuna's verbosity
    if verbose_level >= 1:
        print(f"Optimizing {pipeline.steps[-1][0]} - {n_trials} trials...", end=" ", flush=True)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose_level>=2)
    
    if verbose_level >= 1:
        best_score = study.best_value if study.best_value is not None else float('-inf')
        print(f"Best {scoring}: {best_score:.4f}")

    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    return study.best_params, study.best_value, study

def evaluate_microbiome_models_test(
    dev_df,
    eval_df,
    target_col='BMI',
    age_col='Host age',
    feature_selection_method=None,
    n_features=35,
    n_outer_splits=5,
    n_inner_splits=5,
    n_bootstraps=1000,
    random_state=42,
    use_optuna=False,
    optuna_trials=50,
    skip_tuning=False,
    save_models=True,
    model_save_dir= '../models'
):
    """
    Evaluates models with nested hyperparameter tuning and evaluation set metrics.
    Supports both GridSearchCV and Optuna optimization strategies.
    """

    # Create model directory if it doesn't exist
    if save_models:
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Models will be saved to: {os.path.abspath(model_save_dir)}")


    # --- Data Preparation ---
    X_dev, y_dev, X_eval, y_eval, feature_names = prepare_data(
        dev_df, eval_df, target_col, age_col
    )
    
    # --- Model Configuration ---
    models = configure_models()
    feature_selectors = configure_feature_selectors(n_features)
    
    # --- Evaluation ---
    results = []
    feature_importances = []
    saved_model_paths = {}  # To store paths of saved models
    best_overall_model = None
    best_score = -float('inf')
    best_model_name = None

    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)

    for model_name, model_info in models.items():
        print(f"\n=== Evaluating {model_name} ===")
        
        model_results = evaluate_single_model(
            model_name,
            model_info,
            X_dev,
            y_dev,
            X_eval,
            y_eval,
            feature_names,
            feature_selection_method,
            feature_selectors,
            outer_cv,
            n_inner_splits,
            n_bootstraps,
            random_state,
            use_optuna,
            optuna_trials,
            age_col,
            skip_tuning
        )
        
        results.append(model_results['metrics'])
        current_score = model_results['metrics']['Eval_R2']['mean'] #Using R2 as a metric
        
        # Track best overall model
        if current_score > best_score:
            best_score = current_score
            best_overall_model = model_results['best_model']
            best_model_name = model_name

        if model_results['feature_importance'] is not None:
            feature_importances.append(model_results['feature_importance'])
        
        # Save individual models
        if save_models and 'best_model' in model_results:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{model_name}_{timestamp}.joblib"
                model_path = os.path.join(model_save_dir, model_filename)
                
                joblib.dump(model_results['best_model'], model_path)
                saved_model_paths[model_name] = model_path
                print(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                print(f"Error saving {model_name} model: {str(e)}")
    
    # Save the best overall model
    if save_models and best_overall_model is not None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            winner_filename = f"winner_{best_model_name}_{timestamp}.joblib"
            winner_path = os.path.join(model_save_dir, winner_filename)
            
            joblib.dump(best_overall_model, winner_path)
            saved_model_paths['winner'] = winner_path
            print(f"\n*** Best model is {best_model_name} with R2 score {best_score:.3f} ***")
            print(f"Saved winner model to {winner_path}")
        except Exception as e:
            print(f"Error saving winner model: {str(e)}")
    
    # Combine feature importances into a single DataFrame
    feature_importances_df = format_feature_importances(feature_importances, models)
    
    return pd.DataFrame(results), feature_importances_df, saved_model_paths, best_overall_model

def prepare_data(dev_df, eval_df, target_col, age_col):
    """Prepare and validate input data."""
    dev_filter = EmptySampleFilter(target_col=target_col)
    dev_filter.fit(dev_df)
    filtered_dev = dev_filter.transform(dev_df)
    feature_names = filtered_dev.drop(columns=[target_col]).columns.values
    X_dev = filtered_dev.drop(columns=[target_col]).values
    y_dev = filtered_dev[target_col].values
    
    X_eval = eval_df.drop(columns=[target_col]).values
    y_eval = eval_df[target_col].values
    
    return X_dev, y_dev, X_eval, y_eval, feature_names

def configure_models():
    """Define model configurations with their hyperparameters."""
    return {
        "ElasticNet": {
            "model": ElasticNetCV(max_iter=10000, tol=1e-4, selection='random'),
            "params": {
                'model__l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
                'model__eps': [1e-4, 1e-3, 1e-2]
            },
            "optuna_params": {
                'model__l1_ratio': ('uniform', 0, 1),
                'model__eps': ('log-uniform', 1e-4, 1e-2)
            }
        },
        "BayesianRidge": {
            "model": BayesianRidge(),
            "params": {
                'model__alpha_1': [1e-7, 1e-6, 1e-5],
                'model__alpha_2': [1e-7, 1e-6, 1e-5],
                'model__lambda_1': [1e-7, 1e-6, 1e-5],
                'model__lambda_2': [1e-7, 1e-6, 1e-5]
            },
            "optuna_params": {
                'model__alpha_1': ('log-uniform', 1e-8, 1e-4),
                'model__alpha_2': ('log-uniform', 1e-8, 1e-4),
                'model__lambda_1': ('log-uniform', 1e-8, 1e-4),
                'model__lambda_2': ('log-uniform', 1e-8, 1e-4)
            }
        },
        "SVM": {
            "model": SVR(kernel='rbf'),
            "params": {
                'model__C': np.logspace(-2, 3, 6),
                'model__gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),
                'model__epsilon': [0.01, 0.1, 0.5]
            },
            "optuna_params": {
                'model__C': ('log-uniform', 1e-2, 1e3),
                'model__gamma': ('log-uniform', 1e-3, 1e1),
                'model__epsilon': ('log-uniform', 0.01, 0.5)
            }
        }
    }

def configure_feature_selectors(n_features):
    """Configure available feature selection methods."""
    return {
        None: None,
        'RFE' : RFE(LinearRegression(), n_features_to_select=n_features),
        'variance': VarianceThreshold(threshold=0.05),
        'select_from_model': SelectFromModel(ElasticNetCV(), max_features=n_features)
    }

def build_pipeline(model, feature_selector, age_col):
    """Construct the processing pipeline."""
    steps = [
        ('transform', CompositionalTransformer(exclude_col=age_col)),
        ('scale', StandardScaler())
    ]
    
    if feature_selector is not None:
        steps.append(('feature_selection', clone(feature_selector)))
    
    steps.append(('model', clone(model)))
    return Pipeline(steps)

def optimize_with_optuna(X_train, y_train, pipeline, param_space, n_trials, cv, random_state):
    """Perform hyperparameter optimization using Optuna."""
    print("Using Optuna for hyperparameter optimization...")
    opt_params, opt_score, _ = optimize_optuna(
        X_train, y_train,
        pipeline=pipeline,
        param_space=param_space,
        n_trials=n_trials,
        cv=cv,
        random_state=random_state
    )
    pipeline.set_params(**opt_params)
    return pipeline, opt_params

def optimize_with_gridsearch(X_train, y_train, pipeline, param_grid, cv):
    """Perform hyperparameter optimization using GridSearchCV."""
    print("Using GridSearchCV for hyperparameter optimization...")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def evaluate_single_model(
    model_name,
    model_info,
    X_dev,
    y_dev,
    X_eval,
    y_eval,
    feature_names,
    feature_selection_method,
    feature_selectors,
    outer_cv,
    n_inner_splits,
    n_bootstraps,
    random_state,
    use_optuna,
    optuna_trials,
    age_col,
    skip_tuning
):
    """Evaluate a single model with nested cross-validation."""
    # Track metrics across folds
    dev_metrics = {'R2': [], 'RMSE': [], 'MAE': []}
    eval_metrics = {'R2': [], 'RMSE': [], 'MAE': []}
    best_params = []
    fold_feature_importances = []
    
    selector = feature_selectors.get(feature_selection_method)
    
    for train_idx, val_idx in outer_cv.split(X_dev):
        X_train, X_val = X_dev[train_idx], X_dev[val_idx]
        y_train, y_val = y_dev[train_idx], y_dev[val_idx]
        
        pipeline = build_pipeline(model_info["model"], selector, age_col)
        
        # Hyperparameter tuning
        if skip_tuning:
            # Skip hyperparameter tuning and use default parameters
            pipeline.fit(X_train, y_train)
            params = None
        else:
            if use_optuna and 'optuna_params' in model_info:
                pipeline, params = optimize_with_optuna(
                    X_train, y_train,
                    pipeline=pipeline,
                    param_space=model_info["optuna_params"],
                    n_trials=optuna_trials,
                    cv=n_inner_splits,
                    random_state=random_state
                )
            else:
                pipeline, params = optimize_with_gridsearch(
                    X_train, y_train,
                    pipeline=pipeline,
                    param_grid=model_info["params"],
                    cv=n_inner_splits
                )
        
        best_params.append(params)
        
        # Fit the pipeline (for Optuna case where we didn't fit during optimization)
        if use_optuna:
            pipeline.fit(X_train, y_train)
        
        # Validation on dev fold
        evaluate_fold(pipeline, X_val, y_val, dev_metrics)
        
        # Evaluation on test set
        evaluate_fold(pipeline, X_eval, y_eval, eval_metrics)
        
        # Feature importance analysis
        importance_df = analyze_feature_importance_safe(pipeline, feature_names, model_name)
        if importance_df is not None:
            fold_feature_importances.append(importance_df)
    
    # Aggregate feature importances across folds
    avg_importance = aggregate_feature_importances(fold_feature_importances)
    
    # Train final model
    final_pipeline, most_common_params = train_final_model(
        model_info["model"],
        selector,
        best_params,
        X_dev,
        y_dev,
        age_col
    )
    
    # Bootstrap evaluation
    boot_metrics = bootstrap_evaluate(
        final_pipeline,
        X_eval,
        y_eval,
        n_bootstraps,
        random_state
    )
    
    return {
        'metrics': create_result_dict(
            model_name,
            feature_selection_method,
            dev_metrics,
            eval_metrics,
            boot_metrics,
            most_common_params,
            X_dev,
            X_eval
        ),
        'feature_importance': avg_importance,
        'best_model': final_pipeline 
    }

def evaluate_fold(pipeline, X, y, metrics_store):
    """Evaluate pipeline on a single fold and store metrics."""
    y_pred = pipeline.predict(X)
    metrics_store['R2'].append(r2_score(y, y_pred))
    metrics_store['RMSE'].append(np.sqrt(mean_squared_error(y, y_pred)))
    metrics_store['MAE'].append(mean_absolute_error(y, y_pred))

def analyze_feature_importance_safe(pipeline, feature_names, model_name):
    """Safely compute feature importance with error handling."""
    try:
        importance_df = analyze_feature_importance(pipeline, feature_names)
        return importance_df if not importance_df.empty else None
    except Exception as e:
        print(f"Warning: Could not compute feature importance for {model_name}: {str(e)}")
        return None

def aggregate_feature_importances(fold_feature_importances):
    """Aggregate feature importances across folds."""
    if not fold_feature_importances:
        return None
        
    combined_df = pd.concat(fold_feature_importances)
    avg_importance = combined_df.groupby('feature')['importance'].mean().reset_index()
    return avg_importance.sort_values('importance', ascending=False)

def train_final_model(model, selector, best_params, X_dev, y_dev, age_col):
    """Train the final model on all development data."""
    final_pipeline = build_pipeline(model, selector, age_col)
    
    # Use most frequent best params from CV
    most_common_params = None
    if best_params:
        param_counts = Counter([str(p) for p in best_params])
        if len(param_counts) > 0:
            most_common_params = eval(param_counts.most_common(1)[0][0])
            if most_common_params:
                final_pipeline.set_params(**most_common_params)
    
    final_pipeline.fit(X_dev, y_dev)
    return final_pipeline, most_common_params

def bootstrap_evaluate(pipeline, X_eval, y_eval, n_bootstraps, random_state):
    """Perform bootstrap evaluation of the model."""
    rng = np.random.RandomState(random_state)
    boot_metrics = {'R2': [], 'RMSE': [], 'MAE': []}
    
    for _ in range(n_bootstraps):
        idx = rng.choice(len(X_eval), size=len(X_eval), replace=True)
        X_sample, y_sample = X_eval[idx], y_eval[idx]
        evaluate_fold(pipeline, X_sample, y_sample, boot_metrics)
    
    return boot_metrics

def summarize_metrics(metrics):
    """Calculate summary statistics for metrics."""
    return {
        'mean': np.mean(metrics),
        'median': np.median(metrics),
        'std': np.std(metrics),
        'CI_95': (np.percentile(metrics, 2.5), np.percentile(metrics, 97.5))
    }

def create_result_dict(
    model_name,
    feature_selection_method,
    dev_metrics,
    eval_metrics,
    boot_metrics,
    most_common_params,
    X_dev,
    X_eval
):
    """Create the results dictionary for a single model."""
    return {
        'Model': model_name,
        'Feature_Selection': feature_selection_method or 'None',
        'Dev_R2': summarize_metrics(dev_metrics['R2']),
        'Dev_RMSE': summarize_metrics(dev_metrics['RMSE']),
        'Dev_MAE': summarize_metrics(dev_metrics['MAE']),
        'Eval_R2': summarize_metrics(eval_metrics['R2']),
        'Eval_RMSE': summarize_metrics(eval_metrics['RMSE']),
        'Eval_MAE': summarize_metrics(eval_metrics['MAE']),
        'Boot_R2': summarize_metrics(boot_metrics['R2']),
        'Boot_RMSE': summarize_metrics(boot_metrics['RMSE']),
        'Boot_MAE': summarize_metrics(boot_metrics['MAE']),
        'Best_Params': most_common_params,
        'Dev_Samples': len(X_dev),
        'Eval_Samples': len(X_eval)
    }

def format_feature_importances(feature_importances, models):
    """Format feature importances into a DataFrame."""
    return pd.concat(
        [imp for imp in feature_importances if imp is not None], 
        keys=models.keys(), 
        names=['Model', 'Index']
    ).reset_index(level='Index', drop=True).reset_index()


# ----------------- Helper Functions for detecting overfit and ploting results ----------------- #

class OverfittingAnalyzer:
    """
    A class to analyze and visualize potential overfitting in model evaluation results.
    """
    
    def __init__(self, threshold: float = 0.15):
        """
        Initialize the analyzer with a threshold for overfitting detection.
        
        Args:
            threshold: Relative performance drop threshold (e.g., 0.15 = 15% drop)
        """
        self.threshold = threshold
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure consistent plot styling."""
        sns.set_style("whitegrid")
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def analyze(self, evaluation_results: pd.DataFrame, plot: bool = True) -> Dict:
        """
        Comprehensive overfitting analysis with optional visualization.
        
        Args:
            evaluation_results: Results dataframe from model evaluation
            plot: Whether to generate diagnostic plots
            
        Returns:
            Dictionary containing:
            - overfitting_status: ['severe', 'moderate', 'none']
            - metrics: DataFrame with performance metrics
            - failed_checks: List of failed diagnostic checks
        """
        metrics_df = self._calculate_performance_metrics(evaluation_results)
        analysis = self._perform_diagnostic_checks(metrics_df)
        
        if plot:
            self._generate_diagnostic_plots(metrics_df)
        
        return {
            'overfitting_status': analysis['status'],
            'metrics': metrics_df,
            'failed_checks': analysis['failed_checks']
        }
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics and drops between dev and eval sets.
        
        Args:
            results: Raw evaluation results dataframe
            
        Returns:
            DataFrame with calculated metrics
        """
        metrics = []
        
        for _, row in results.iterrows():
            dev_r2 = row['Dev_R2']['mean']
            eval_r2 = row['Eval_R2']['mean']
            r2_drop = self._calculate_relative_drop(dev_r2, eval_r2)
            
            dev_rmse = row['Dev_RMSE']['mean']
            eval_rmse = row['Eval_RMSE']['mean']
            rmse_increase = self._calculate_relative_increase(dev_rmse, eval_rmse)
            
            metrics.append({
                'Model': row['Model'],
                'Feature_Selection': row['Feature_Selection'],
                'Dev_R2': dev_r2,
                'Eval_R2': eval_r2,
                'R2_Drop_Pct': r2_drop * 100,
                'Dev_RMSE': dev_rmse,
                'Eval_RMSE': eval_rmse,
                'RMSE_Increase_Pct': rmse_increase * 100,
                'Dev_Eval_Ratio': self._safe_divide(dev_r2, eval_r2),
                'Overfitting_Risk': self._assess_risk_level(r2_drop)
            })
        
        return pd.DataFrame(metrics)
    
    def _calculate_relative_drop(self, dev_value: float, eval_value: float) -> float:
        """Calculate relative performance drop between dev and eval sets."""
        return (dev_value - eval_value) / dev_value if dev_value != 0 else np.inf
    
    def _calculate_relative_increase(self, dev_value: float, eval_value: float) -> float:
        """Calculate relative performance increase between dev and eval sets."""
        return (eval_value - dev_value) / dev_value if dev_value != 0 else np.inf
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division with protection against division by zero."""
        return numerator / max(denominator, 1e-6)
    
    def _assess_risk_level(self, r2_drop: float) -> str:
        """Determine overfitting risk level based on R² drop."""
        if r2_drop > self.threshold:
            return 'High'
        elif r2_drop > self.threshold / 2:
            return 'Medium'
        return 'Low'
    
    def _perform_diagnostic_checks(self, metrics_df: pd.DataFrame) -> Dict:
        """
        Perform diagnostic checks for overfitting patterns.
        
        Args:
            metrics_df: DataFrame with calculated metrics
            
        Returns:
            Dictionary with analysis results
        """
        failed_checks = []
        
        # Check 1: Large R² drop
        if any(metrics_df['R2_Drop_Pct'] > self.threshold * 100):
            failed_checks.append(f"R² drop > {self.threshold*100}% detected")
        
        # Check 2: Negative evaluation R²
        if any(metrics_df['Eval_R2'] < 0):
            failed_checks.append("Negative evaluation R² (worse than baseline)")
        
        # Check 3: Dev-to-eval ratio
        if any(metrics_df['Dev_Eval_Ratio'] > 3):
            failed_checks.append("Dev-to-eval ratio > 3:1")
        
        # Determine overall status
        if len(failed_checks) >= 2:
            status = 'severe'
        elif len(failed_checks) == 1:
            status = 'moderate'
        else:
            status = 'none'
        
        return {'status': status, 'failed_checks': failed_checks}
    
    def _generate_diagnostic_plots(self, metrics_df: pd.DataFrame):
        """Generate diagnostic plots for overfitting analysis."""
        plt.figure(figsize=(16, 6))
        
        # R² comparison plot
        plt.subplot(1, 2, 1)
        self._plot_r2_comparison(metrics_df)
        
        # Performance gap plot
        plt.subplot(1, 2, 2)
        self._plot_performance_gaps(metrics_df)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_r2_comparison(self, metrics_df: pd.DataFrame):
        """Plot R² comparison between dev and eval sets."""
        ax = sns.barplot(
            data=metrics_df.melt(
                id_vars=['Model', 'Feature_Selection'], 
                value_vars=['Dev_R2', 'Eval_R2']
            ),
            x='Model', 
            y='value', 
            hue='variable',
            hue_order=['Dev_R2', 'Eval_R2']
        )
        
        plt.title('R² Comparison: Dev vs Eval')
        plt.ylim(metrics_df[['Dev_R2', 'Eval_R2']].min().min() - 0.1, 1)
        plt.axhline(0, color='k', linestyle='--')
        plt.xticks(rotation=45)
    
    def _plot_performance_gaps(self, metrics_df: pd.DataFrame):
        """Plot performance gaps with thresholds."""
        ax = sns.barplot(
            data=metrics_df,
            x='Model',
            y='R2_Drop_Pct',
            hue='Feature_Selection',
            palette='Reds_r'
        )
        
        plt.title('Percentage R² Drop from Dev to Eval')
        plt.axhline(self.threshold * 100, color='r', linestyle='--', label='Severe Threshold')
        plt.axhline(self.threshold * 50, color='orange', linestyle='--', label='Moderate Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        
        # Annotate bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )

def plot_model_results_final(results_df):
    """
    Visualize model comparison results with MAE, R², and RMSE metrics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_microbiome_models()
    """
    # Setup plot configuration
    _setup_plot_style()
    
    # Create figure and plot all components
    fig = plt.figure(figsize=(20, 12))
    _plot_performance_metrics(fig, results_df)
    _plot_sample_sizes_and_feature_impact(fig, results_df)
    
    plt.tight_layout()
    plt.show()
    
    # Display detailed metrics
    _display_detailed_metrics(results_df)

def _setup_plot_style():
    """Configure consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

def _plot_performance_metrics(fig, results_df):
    """
    Plot the performance metrics (R2, RMSE, MAE) comparison between development and evaluation.
    
    Args:
        fig: matplotlib Figure object
        results_df: DataFrame containing model results
    """
    metrics_config = [
        ('R2', 'upper left', (-0.1, 1.0)),
        ('RMSE', 'upper right', (0, None)),
        ('MAE', 'upper right', (0, None))
    ]
    
    palette = sns.color_palette("husl", len(results_df))
    
    for idx, (metric, legend_pos, ylim) in enumerate(metrics_config, 1):
        ax = fig.add_subplot(2, 2, idx)
        _plot_single_metric(ax, results_df, metric, palette)
        
        # Configure plot appearance
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Development (CV)', 'Evaluation'])
        ax.set_ylabel(metric)
        ax.set_title(f'Model Performance Comparison ({metric})')
        ax.legend(bbox_to_anchor=(1, 1), loc=legend_pos)
        ax.set_ylim(ylim)

def _plot_single_metric(ax, results_df, metric, palette):
    """
    Plot a single metric comparison between development and evaluation.
    
    Args:
        ax: matplotlib Axes object
        results_df: DataFrame containing model results
        metric: metric to plot (R2, RMSE, or MAE)
        palette: color palette to use
    """
    for i, (_, row) in enumerate(results_df.iterrows()):
        dev_stats = row[f'Dev_{metric}']
        eval_stats = row[f'Eval_{metric}']
        
        # Calculate error bars
        yerr_lower = [
            dev_stats['median'] - dev_stats['CI_95'][0],
            eval_stats['median'] - eval_stats['CI_95'][0]
        ]
        yerr_upper = [
            dev_stats['CI_95'][1] - dev_stats['median'],
            eval_stats['CI_95'][1] - eval_stats['median']
        ]
        
        ax.errorbar(
            [0, 1], 
            [dev_stats['median'], eval_stats['median']],
            yerr=[yerr_lower, yerr_upper],
            fmt='o-',
            color=palette[i],
            label=f"{row['Model']} ({row['Feature_Selection']})",
            capsize=5,
            markersize=8
        )

def _plot_sample_sizes_and_feature_impact(fig, results_df):
    """
    Plot sample sizes and feature selection impact.
    
    Args:
        fig: matplotlib Figure object
        results_df: DataFrame containing model results
    """
    ax = fig.add_subplot(2, 2, 4)
    
    # Plot sample sizes
    _plot_sample_sizes(ax, results_df)
    
    # Plot feature selection impact if multiple methods exist
    if results_df['Feature_Selection'].nunique() > 1:
        _plot_feature_selection_impact(ax, results_df)
    
    ax.set_title('Sample Sizes and Feature Selection Impact')

def _plot_sample_sizes(ax, results_df):
    """
    Plot development and evaluation sample sizes as bars.
    
    Args:
        ax: matplotlib Axes object
        results_df: DataFrame containing model results
    """
    x_pos = np.arange(len(results_df))
    width = 0.4
    
    ax.bar(
        x=x_pos - width/2,
        height=results_df['Dev_Samples'],
        width=width,
        color='skyblue',
        alpha=0.6,
        label='Development'
    )
    ax.bar(
        x=x_pos + width/2,
        height=results_df['Eval_Samples'],
        width=width,
        color='salmon',
        alpha=0.6,
        label='Evaluation'
    )
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'])
    ax.set_ylabel('Number of Samples')
    ax.legend(loc='upper left')

def _plot_feature_selection_impact(ax, results_df):
    """
    Plot feature selection impact on R2 scores.
    
    Args:
        ax: matplotlib Axes object
        results_df: DataFrame containing model results
    """
    ax2 = ax.twinx()
    
    plot_data = pd.DataFrame({
        'Model': results_df['Model'],
        'R2': results_df['Eval_R2'].apply(lambda x: x['median']),
        'Method': results_df['Feature_Selection']
    })
    
    markers = ['o', 's', 'D', '^'][:results_df['Feature_Selection'].nunique()]
    
    sns.pointplot(
        data=plot_data,
        x='Model',
        y='R2',
        hue='Method',
        palette='dark',
        ax=ax2,
        join=False,
        markers=markers
    )
    
    ax2.set_ylabel('Evaluation R² (median)')
    ax2.legend(bbox_to_anchor=(1, 1))

def _display_detailed_metrics(results_df):
    """
    Display detailed performance metrics in formatted tables.
    
    Args:
        results_df: DataFrame containing model results
    """
    print("\n=== Detailed Performance Metrics ===")
    
    for metric in ['R2', 'RMSE', 'MAE']:
        print(f"\n{metric.upper()} Scores (median with 95% CI):")
        
        display_df = results_df[[
            'Model', 
            'Feature_Selection',
            f'Dev_{metric}',
            f'Eval_{metric}'
        ]].copy()
        
        # Format metric display
        display_df[f'Dev_{metric}'] = display_df[f'Dev_{metric}'].apply(
            lambda x: f"{x['median']:.3f} ({x['CI_95'][0]:.3f}-{x['CI_95'][1]:.3f})"
        )
        display_df[f'Eval_{metric}'] = display_df[f'Eval_{metric}'].apply(
            lambda x: f"{x['median']:.3f} ({x['CI_95'][0]:.3f}-{x['CI_95'][1]:.3f})"
        )
        
        # Display with styling
        styled_df = display_df.style.set_properties(**{
            'text-align': 'center',
        })
        display(styled_df)