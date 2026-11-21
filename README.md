NeuroMaintain: Intelligent Machine Failure Prediction

An artificial neural network (ANN) that estimates the probability of a machine failure from sensor readings, with an interactive Streamlit app.

Live app:https://neuromaintain-8gjh3phskuhjzvc3xvswn3.streamlit.app/

Disclaimer: This is a learning project. The output is a model estimate, not a guarantee of machine condition


Problem

Unplanned machine failures cause downtime and repair costs. Predictive maintenance uses sensor data to flag machines that are likely to fail, so maintenance can be scheduled before a breakdown happens.

Features
Takes 6 inputs: air temperature, process temperature, rotational speed, torque, tool wear and machine type
Returns the failure probability (not just yes or no)
Shows a clear status: "No Failure Predicted" or "Failure Predicted"
Simple Streamlit interface with default values
Dataset

AI4I 2020 Predictive Maintenance Dataset (UCI): 10,000 records of machine sensor readings with a failure label.

Feature	Description
Air temperature [K]	Ambient temperature
Process temperature [K]	Temperature during the process
Rotational speed [rpm]	Spindle speed
Torque [Nm]	Torque applied
Tool wear [min]	Minutes of tool use
Type	Machine quality variant (L, M or H)
Machine failure	Target (0 or 1)

Failures are rare (about 3 to 4% of records), so the data is highly imbalanced. Accuracy alone is misleading here, because a model that always predicts "no failure" would still score about 96%.

Approach
Load and clean the data, encode the machine type
Scale numeric features with a scaler (saved as neuromaintain_scaler.pkl)
Split into train and test sets
Build an ANN with Keras (dense layers with ReLU and a sigmoid output)
Handle class imbalance [FILL IN: class weights / SMOTE / other]
Evaluate with precision, recall and F1 on the failure class, not only accuracy
Save the model (neuromaintain_model.keras) and deploy with Streamlit
Results

Fill these in from your test-set classification report:

Metric (failure class)	Value
Precision	[FILL IN]
Recall	[FILL IN]
F1-score	[FILL IN]
Overall accuracy	[FILL IN]

Baseline comparison: [FILL IN: e.g. Random Forest recall X vs ANN recall Y, and which one performed better].

Limitations
Trained on a synthetic dataset, so real factory performance is untested
Failures are rare, so the failure class has few examples
The decision threshold (default 0.5) can be tuned depending on how costly missed failures are
No explanation of which input drove a prediction
Tech stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Pandas, Streamlit.

Run locally
bash
git clone https://github.com/shrutiyadav1/neuromaintain.git
cd neuromaintain
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
Project structure
neuromaintain/
├── app.py                          # Streamlit app
├── neuromaintain_model.keras       # Trained ANN
├── neuromaintain_scaler.pkl        # Fitted scaler
├── requirements.txt
└── README.md
Future improvements
Tune the decision threshold for higher recall
Add SHAP explanations for each prediction
Add a warning level between "no failure" and "failure"
Extend to time-series data with an LSTM for remaining useful life prediction
Author

Shruti Yadav, B.Tech CSE (AI) GitHub | LinkedIn
