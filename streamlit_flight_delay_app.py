import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LassoFlightDelayPredictor:
    def __init__(self):
        self.model = None
        self.df = None
        self.X = None
        self.y = None
        self.preprocessor = None
        self.train_idx = None
        self.test_idx = None
        self.results = {}

    def load_and_preprocess_data(self, file_path):
        logging.info("Loading data from %s...", file_path)
        df = pd.read_csv(file_path)
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        df = df[df['DepDelay'].between(-60, 1440)]
        df = df.dropna(subset=['DepDelay'])
        self.df = df.reset_index(drop=True)
        logging.info(f"After cleaning: {self.df.shape}")
        return self.df

    def filter_by_airline(self, airline_code):
        """Filter data by airline code (Origin or Dest)"""
        if airline_code:
            # Filter flights where the airline appears in Origin or Dest
            mask = (self.df['Origin'] == airline_code) | (self.df['Dest'] == airline_code)
            self.df = self.df[mask].reset_index(drop=True)
            logging.info(f"Filtered to {len(self.df)} flights for airline {airline_code}")
        return self.df

    def engineer_features(self):
        logging.info("Engineering features...")
        df = self.df
        df['hour'] = df['CRSDepTime'] // 100
        df['minute'] = df['CRSDepTime'] % 100
        df['departure_time_minutes'] = df['hour'] * 60 + df['minute']
        df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Early_Morning', 'Morning', 'Afternoon', 'Evening'], right=False, include_lowest=True)
        if 'FlightDate' in df.columns:
            df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
            df['day_of_week'] = df['FlightDate'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        for col in ['precip_mm', 'cloud_pct', 'wind_speed_mps']:
            if col not in df.columns:
                df[col] = 0
        df['weather_severity'] = (
            (df['precip_mm'] > 5).astype(int) +
            (df['cloud_pct'] > 80).astype(int) +
            (df['wind_speed_mps'] > 15).astype(int)
        )
        if 'Distance' in df.columns:
            df['route_complexity'] = pd.cut(df['Distance'], bins=[0, 500, 1000, 2000, 5000], labels=['Short', 'Medium', 'Long', 'Very_Long'])
        else:
            df['route_complexity'] = 'Unknown'
        for col in ['WeatherDelay', 'LateAircraftDelay']:
            if col not in df.columns:
                df[col] = 0
        df['has_weather_delay'] = (df['WeatherDelay'] > 0).astype(int)
        df['has_aircraft_delay'] = (df['LateAircraftDelay'] > 0).astype(int)
        if 'temperature_c' in df.columns:
            df['temp_category'] = pd.cut(df['temperature_c'], bins=[-50, 0, 10, 20, 50], labels=['Very_Cold', 'Cold', 'Mild', 'Warm'])
        else:
            df['temp_category'] = 'Unknown'
        self.df = df
        logging.info("Feature engineering completed!")
        return self.df

    def prepare_features(self):
        logging.info("Preparing features...")
        df = self.df
        categorical_features = [
            'Origin', 'Dest', 'time_of_day', 'route_complexity',
            'temp_category', 'has_weather_delay', 'has_aircraft_delay'
        ]
        numerical_features = [
            'departure_time_minutes', 'Distance', 'temperature_c',
            'precip_mm', 'cloud_pct', 'wind_speed_mps',
            'LateAircraftDelay', 'WeatherDelay'
        ]
        if 'day_of_week' in df.columns:
            numerical_features += ['day_of_week', 'is_weekend']
        categorical_features = [col for col in categorical_features if col in df.columns]
        numerical_features = [col for col in numerical_features if col in df.columns]
        logging.info(f"Categorical features: {categorical_features}")
        logging.info(f"Numerical features: {numerical_features}")
        X = df[categorical_features + numerical_features].copy()
        y = df['DepDelay'].copy()
        for col in categorical_features:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        for col in numerical_features:
            X[col] = X[col].fillna(X[col].median())
        self.X, self.y = X, y
        return self.X, self.y

    def create_preprocessing_pipeline(self):
        X = self.X
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.preprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
        return self.preprocessor

    def train_lasso(self):
        logging.info("Training Lasso model with hyperparameter tuning (GridSearchCV)...")
        if len(self.df) < 100:
            st.error(f"Not enough data for airline. Only {len(self.df)} flights found. Please try a different airline.")
            return None
            
        self.train_idx, self.test_idx = train_test_split(
            self.df.index, test_size=0.2, random_state=42
        )
        X_train, X_test = self.X.loc[self.train_idx], self.X.loc[self.test_idx]
        y_train, y_test = self.y.loc[self.train_idx], self.y.loc[self.test_idx]
        preprocessor = self.create_preprocessing_pipeline()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(max_iter=10000))
        ])
        # Grid search for best alpha
        param_grid = {'regressor__alpha': np.logspace(-3, 2, 20)}
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        best_alpha = grid.best_params_['regressor__alpha']
        y_pred = best_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='r2')
        self.model = best_pipeline
        self.results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': y_pred,
            'best_alpha': best_alpha
        }
        logging.info(f"Best Lasso alpha: {best_alpha}")
        logging.info(f"Lasso - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        return self.results

    def create_visualizations(self):
        """Create all visualizations for Streamlit"""
        if not self.results:
            return None
            
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        y_test = self.y.loc[self.test_idx]
        y_pred = self.results['predictions']
        ax1.scatter(y_test, y_pred, alpha=0.5, color='green')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Delay (minutes)')
        ax1.set_ylabel('Predicted Delay (minutes)')
        ax1.set_title('Actual vs Predicted Delays (Lasso)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Delay (minutes)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Delay distribution
        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(self.y, bins=50, kde=True, color='blue', ax=ax3)
        ax3.set_xlabel('Departure Delay (minutes)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Departure Delays')
        ax3.axvline(self.y.mean(), color='red', linestyle='--', label=f'Mean: {self.y.mean():.1f}')
        ax3.legend()
        
        delays_filtered = self.y[(self.y >= -30) & (self.y <= 120)]
        sns.histplot(delays_filtered, bins=30, kde=True, color='green', ax=ax4)
        ax4.set_xlabel('Departure Delay (minutes)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Delays (-30 to 120 minutes)')
        
        # 4. Feature importance (if available)
        fig4, ax5 = plt.subplots(figsize=(12, 8))
        if hasattr(self.model.named_steps['regressor'], 'coef_'):
            # Get feature names after preprocessing
            feature_names = []
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                feature_names = [f'Feature_{i}' for i in range(len(self.model.named_steps['regressor'].coef_))]
            
            coefs = self.model.named_steps['regressor'].coef_
            # Get top 15 features by absolute coefficient value
            top_indices = np.argsort(np.abs(coefs))[-15:]
            top_coefs = coefs[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            colors = ['red' if x < 0 else 'blue' for x in top_coefs]
            ax5.barh(range(len(top_coefs)), top_coefs, color=colors)
            ax5.set_yticks(range(len(top_coefs)))
            ax5.set_yticklabels(top_names)
            ax5.set_xlabel('Coefficient Value')
            ax5.set_title('Top 15 Feature Coefficients (Lasso)')
            ax5.grid(True, alpha=0.3)
        
        return fig1, fig2, fig3, fig4

    def generate_report(self):
        """Generate a comprehensive report"""
        if not self.results:
            return "No results available. Please train the model first."
            
        report = f"""
## Flight Delay Prediction Model Report

### Dataset Information
- **Total flights analyzed**: {len(self.df):,}
- **Features used**: {len(self.X.columns)}
- **Average delay**: {self.y.mean():.2f} minutes
- **Delay standard deviation**: {self.y.std():.2f} minutes

### Model Performance
- **Best Lasso alpha**: {self.results['best_alpha']:.4f}
- **R² Score**: {self.results['r2']:.3f}
- **RMSE**: {self.results['rmse']:.2f} minutes
- **MAE**: {self.results['mae']:.2f} minutes
- **Cross-Validation R²**: {self.results['cv_r2_mean']:.3f} (±{self.results['cv_r2_std'] * 2:.3f})

### Interpretation
- The model explains **{self.results['r2']*100:.1f}%** of the variance in departure delays
- On average, predictions are off by **{self.results['mae']:.1f}** minutes
- Cross-validation suggests the model generalizes well to unseen data
        """
        return report

def main():
    st.set_page_config(
        page_title="Flight Delay Predictor",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("✈️ Flight Delay Prediction Model")
    st.markdown("---")
    
    # Sidebar for airline selection
    st.sidebar.header("🔧 Model Configuration")
    
    # Load data
    try:
        predictor = LassoFlightDelayPredictor()
        df = predictor.load_and_preprocess_data("merged_flight_weather.csv")
        
        # Get unique airlines
        all_airlines = sorted(set(df['Origin'].unique()) | set(df['Dest'].unique()))
        
        # Airline selection
        selected_airline = st.sidebar.selectbox(
            "Select Airline Code:",
            options=['All Airlines'] + all_airlines,
            help="Choose an airline code to analyze specific airline data, or 'All Airlines' for the full dataset"
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
        show_report = st.sidebar.checkbox("Show Detailed Report", value=True)
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Processing data and training model..."):
                # Filter by airline if selected
                if selected_airline != 'All Airlines':
                    df = predictor.filter_by_airline(selected_airline)
                    st.info(f"Analyzing {len(df)} flights for airline: {selected_airline}")
                else:
                    st.info(f"Analyzing all {len(df)} flights in the dataset")
                
                # Check if we have enough data
                if len(df) < 50:
                    st.error(f"Not enough data for analysis. Only {len(df)} flights found.")
                    st.stop()
                
                # Process data
                df = predictor.engineer_features()
                X, y = predictor.prepare_features()
                
                # Train model
                results = predictor.train_lasso()
                
                if results is None:
                    st.error("Model training failed. Please try with a different airline or check your data.")
                    st.stop()
                
                # Display results
                st.success("✅ Analysis completed successfully!")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{results['r2']:.3f}")
                with col2:
                    st.metric("RMSE", f"{results['rmse']:.1f} min")
                with col3:
                    st.metric("MAE", f"{results['mae']:.1f} min")
                with col4:
                    st.metric("CV R²", f"{results['cv_r2_mean']:.3f}")
                
                # Detailed report
                if show_report:
                    st.markdown("---")
                    st.markdown(predictor.generate_report())
                
                # Visualizations
                if show_visualizations:
                    st.markdown("---")
                    st.subheader("📊 Model Visualizations")
                    
                    figs = predictor.create_visualizations()
                    if figs:
                        fig1, fig2, fig3, fig4 = figs
                        
                        # Actual vs Predicted
                        st.subheader("Actual vs Predicted Delays")
                        st.pyplot(fig1)
                        
                        # Residuals
                        st.subheader("Residuals Analysis")
                        st.pyplot(fig2)
                        
                        # Delay distributions
                        st.subheader("Delay Distributions")
                        st.pyplot(fig3)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        st.pyplot(fig4)
                        
                        plt.close('all')  # Close all figures to free memory
                
                # Data summary
                st.markdown("---")
                st.subheader("📋 Data Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Overview:**")
                    st.write(f"- Total flights: {len(df):,}")
                    st.write(f"- Features: {len(X.columns)}")
                    st.write(f"- Average delay: {y.mean():.1f} minutes")
                    st.write(f"- Median delay: {y.median():.1f} minutes")
                
                with col2:
                    st.write("**Delay Statistics:**")
                    st.write(f"- Min delay: {y.min():.1f} minutes")
                    st.write(f"- Max delay: {y.max():.1f} minutes")
                    st.write(f"- Std deviation: {y.std():.1f} minutes")
                    st.write(f"- On-time flights: {len(y[y <= 0]):,} ({(len(y[y <= 0])/len(y)*100):.1f}%)")
    
    except FileNotFoundError:
        st.error("❌ Data file 'merged_flight_weather.csv' not found. Please ensure the file is in the same directory as this app.")
        st.info("💡 Make sure you have the required data file: merged_flight_weather.csv")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Please check your data file and try again.")

if __name__ == "__main__":
    main() 