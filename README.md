Crop Production Prediction Project
This project uses Machine Learning (ML) to predict how many tons of crops will be produced in India based on factors like the crop type, variety, state, season, cost, and farming zone.  
PY
This project was built as part of a 6-week internship with upskill Campus and The IoT Academy in collaboration with UniConverge Technologies Pvt Ltd (UCT).  
DOCX
📂 Project Files
Instead of putting all code into one giant notebook, the project is broken down into simple, clean Python files:  
PY
+ 3
data_pipeline.py: Loads the raw data, cleans up empty spaces, and prepares the datasets.  
PY
custom_transformers.py: A helper file that safely converts text columns (like states and crop names) into numbers that the machine learning models can understand.  
PY
train_model.py: Trains different ML models (like Linear Regression and Random Forest), compares their scores, and saves the best-performing model.  
PY
predict.py: Loads the saved model and uses it to predict crop production for new inputs.  
PY
main.py: The master file that runs everything in order with just one click.  
PY
🛠️ How to Run the Project
Follow these simple steps to run the project on your computer:
Step 1: Install Python Libraries
Make sure you have Python installed, then install the required libraries:
Bash
pip install pandas scikit-learn joblib
Step 2: Run the Whole Project
You don't need to run every file individually. Just run the main.py file, and it will automatically clean the data, train the models, and run a test prediction:  
PY
Bash
python main.py
📝 Example Prediction Output
When you run the prediction test, the script takes a simple input like this:  
PY
Python
sample_input = {
    'crop': 'Wheat',
    'variety': 'Sharbati',
    'state': 'Punjab',
    'season': 'Rabi',
    'cost': 22000.0,
    'recommended_zone': 'Northern Plains'
}
And outputs the predicted result:  
PY
Plaintext
→ Predicted Crop Production: 1485.40 Tons
