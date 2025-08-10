✈ United Airlines Flight Delay Prediction
A machine learning application that predicts departure delays for United Airlines flights using historical flight data and weather conditions — providing accurate, data-driven insights for passengers, airlines, and airports.

🌐 Live Demo
🔗 View Demo

📊 Model Overview
Metric	Value
R² (Accuracy)	73.9%
RMSE	7.04 min
MAE	4.69 min
Dataset Size	69,827 flights
Features	77 engineered

🚀 Key Features
Model Capabilities
📍 Route Analysis – Historical delay patterns by origin/destination

🌦 Weather Integration – Impact of temperature, precipitation, and wind speed

🕒 Time Patterns – Peak hours, weekends, seasonal variations

🛫 Hub Connections – Performance at United Airlines hubs (ORD, DEN, IAH, etc.)

⚙ Operational Factors – Flight duration, distance, and delay type analysis

Demo Website
📝 Interactive flight input form

⚡ Real-time delay predictions

🌦 Weather condition inputs

📱 Mobile-responsive design

🎨 Color-coded results by delay severity

🛠 Technical Stack
Backend (Model)

Python (Machine Learning & Feature Engineering)

Scikit-learn (RandomForest & GradientBoosting)

Pandas / NumPy

Frontend (Demo)

HTML5 / CSS3 / JavaScript

Responsive design & animations

📁 Project Structure
bash
Copy
Edit
flight-delay-predictor/
├── index.html                   # Demo website
├── README.md                    # Project documentation
├── united_airlines_enhanced_model.py  # Core ML model
├── simple_flow_diagram.py       # Flow diagram generator
├── model_flow_diagram.py        # Detailed diagram
├── requirements.txt             # Python dependencies
└── data/
    ├── united_airlines_flights.csv
    └── merged_flight_weather.csv
📈 Feature Engineering
Time-Based

Departure/arrival time in minutes

Peak hour & red-eye flags

Day of week, seasonal patterns

Weather

Temperature ranges & extremes

Precipitation levels/types

Wind speed & visibility

Weather severity index

Route

Distance & complexity

Hub connections

Historical route delays

Operational

Flight duration

Delay type breakdown

Historical carrier performance

🎯 Use Cases
Passengers

Plan airport arrival times

Choose routes with lower delay risk

Airlines

Optimize scheduling and crew allocation

Proactive delay notifications

Airports

Improve gate management & staffing

Enhance passenger services

🔮 Future Enhancements
📡 Real-time weather API integration

✈ Multi-airline support

🧠 Deep learning models

📊 Time series modeling

🎯 External event impact analysis

👥 Team
Mitchell Chen – Team Lead, Machine Learning Engineer
Developed models, algorithms, and overall architecture

Jadryan Pena – Data Engineer
Data cleaning and preprocessing

Abraham Yarba – Market Analyst & Technical Writer
Market research and project documentation
