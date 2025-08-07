# United Airlines Flight Delay Prediction Model

A machine learning model that predicts departure delays for United Airlines flights using historical flight data and weather conditions.

## 🌐 Live Demo

Visit the live demo: [https://mitchellchen03.github.io/Flight_Ai/)

## 📊 Model Performance

- **Accuracy (R²)**: 73.9%
- **RMSE**: 7.04 minutes
- **MAE**: 4.69 minutes
- **Dataset**: 69,827 United Airlines flights
- **Features**: 77 engineered features

## 🚀 Features

### Model Capabilities
- **Route Analysis**: Historical delay patterns by origin/destination
- **Weather Integration**: Temperature, precipitation, wind speed impact
- **Time Patterns**: Peak hours, weekends, seasonal variations
- **Hub Connections**: Major United Airlines hub analysis
- **Operational Factors**: Flight duration, distance, delay types

### Demo Website Features
- **Interactive Form**: Easy-to-use flight information input
- **Real-time Predictions**: Instant delay estimates
- **Weather Integration**: Current weather condition inputs
- **Responsive Design**: Works on desktop and mobile
- **Visual Feedback**: Color-coded results based on delay severity

## 🛠️ Technical Stack

### Backend (Model)
- **Python**: Core machine learning implementation
- **Scikit-learn**: RandomForest and GradientBoosting models
- **Pandas**: Data manipulation and feature engineering
- **NumPy**: Numerical computations

### Frontend (Demo)
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with gradients and animations
- **JavaScript**: Interactive form handling and predictions
- **Responsive Design**: Mobile-first approach

## 📁 Project Structure

```
flight-delay-predictor/
├── index.html              # Main demo website
├── README.md               # This file
├── united_airlines_enhanced_model.py  # Core ML model
├── simple_flow_diagram.py  # Flow diagram generator
├── model_flow_diagram.py   # Detailed flow diagram
├── requirements.txt        # Python dependencies
└── data/                  # Data files (not included in repo)
    ├── united_airlines_flights.csv
    └── merged_flight_weather.csv
```

## 🚀 Hosting on GitHub Pages

### Step 1: Create GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `united-airlines-delay-predictor`
3. Make it public (required for free GitHub Pages)

### Step 2: Upload Files
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/united-airlines-delay-predictor.git
   cd united-airlines-delay-predictor
   ```

2. Copy the files to the repository:
   - `index.html` (main website)
   - `README.md` (this file)
   - Any other supporting files

3. Commit and push:
   ```bash
   git add .
   git commit -m "Initial commit: United Airlines delay predictor demo"
   git push origin main
   ```

### Step 3: Enable GitHub Pages
1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll down to **Pages** section
4. Under **Source**, select **Deploy from a branch**
5. Choose **main** branch and **/(root)** folder
6. Click **Save**

### Step 4: Access Your Website
- Your website will be available at: `https://yourusername.github.io/united-airlines-delay-predictor/`
- It may take a few minutes to deploy

## 🔧 Customization

### Adding Real Model Integration
To connect the demo to your actual trained model:

1. **Deploy Model API**: Use services like:
   - Heroku
   - AWS Lambda
   - Google Cloud Functions
   - Azure Functions

2. **Update JavaScript**: Replace the `simulatePrediction()` function with API calls:

```javascript
async function getPrediction(flightData) {
    const response = await fetch('https://your-api-endpoint.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(flightData)
    });
    return await response.json();
}
```

### Styling Customization
- Modify colors in the CSS variables
- Update the gradient backgrounds
- Change fonts and spacing
- Add your company logo

## 📈 Model Details

### Feature Engineering
The model uses 77 engineered features including:

**Time-based Features:**
- Departure/arrival time in minutes
- Time of day categories (peak hours, red-eye)
- Day of week and seasonal patterns

**Weather Features:**
- Temperature categories and extreme conditions
- Precipitation levels and types
- Wind speed and visibility conditions
- Weather severity index

**Route Features:**
- Distance and route complexity
- Hub connections (ORD, DEN, IAH, etc.)
- Historical delay patterns by route

**Operational Features:**
- Flight duration and scheduled times
- Delay type analysis (carrier, weather, aircraft)
- Historical performance metrics

### Model Selection
- **RandomForest**: 200 trees, max depth 20 (Best performing)
- **GradientBoosting**: 200 trees, max depth 6
- **Cross-validation**: 5-fold validation
- **Feature importance**: Automatic feature ranking

## 🎯 Use Cases

### For Passengers
- **Trip Planning**: Know expected delays in advance
- **Airport Timing**: Plan arrival times more accurately
- **Alternative Routes**: Choose flights with lower delay probability

### For Airlines
- **Operational Planning**: Better resource allocation
- **Customer Communication**: Proactive delay notifications
- **Route Optimization**: Identify problematic routes and times

### For Airports
- **Gate Management**: Optimize gate assignments
- **Staff Scheduling**: Better resource planning
- **Passenger Services**: Improved customer experience

## 📊 Performance Metrics

### Model Accuracy
- **R² = 0.739**: Explains 73.9% of delay variance
- **RMSE = 7.04 min**: Average prediction error
- **MAE = 4.69 min**: Median prediction error

### Key Insights
- **Delay type information** is the strongest predictor
- **Operational factors** matter more than weather
- **Time-based patterns** are moderately important
- **26.1% unexplained variance** due to random factors

## 🔮 Future Enhancements

### Planned Improvements
1. **Real-time Weather**: Live weather API integration
2. **More Airlines**: Extend to other carriers
3. **Deep Learning**: Neural networks for complex patterns
4. **Time Series**: Consider temporal dependencies
5. **External Factors**: Events, holidays, economic indicators

### Deployment Options
1. **API Service**: Real-time predictions
2. **Mobile App**: Passenger notifications
3. **Dashboard**: Operational monitoring
4. **Integration**: Existing airline systems



