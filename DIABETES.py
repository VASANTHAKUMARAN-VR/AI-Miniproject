!pip install scikit-learn matplotlib

# Step 2: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 3: Load the Dataset
df = pd.read_csv('/content/drive/MyDrive/diabetes.csv')

# Step 4: Preprocess the Data
X = df.drop(columns='Outcome')  # Features (independent variables)
y = df['Outcome']  # Target variable (dependent variable - 0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Set up Hyperparameter Tuning for SVM using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Create an SVM model
svm_model = SVC()

# GridSearchCV to tune the hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=3, cv=5)

# Step 6: Train the Model with the Best Parameters
grid_search.fit(X_train_scaled, y_train)

# Best parameters from the Grid Search
print(f"Best parameters found: {grid_search.best_params_}")

# Step 7: Evaluate the Model
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_scaled)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred_test)
print(f"SVM Model Accuracy with Hyperparameter Tuning: {accuracy:.4f}")

# Confusion matrix to visualize prediction performance
cm = confusion_matrix(y_test, y_pred_test)
print(f"Confusion Matrix:\n{cm}")

# Classification report (includes precision, recall, F1-score)
report = classification_report(y_test, y_pred_test)
print(f"Classification Report:\n{report}")

# Step 8: Function to Predict User Input and Diabetes Risk Over 5 Years
def predict_user_input_and_risk():
    print("Please enter the following values:")

    # Get input from the user
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    # Create a DataFrame for the user's input
    user_input = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Standardize user input using the same scaler as the training data
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction
    prediction = best_model.predict(user_input_scaled)

    # Output the result
    result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
    print(f"\nThe model predicts: {result}")
    print(f"Prediction value (0 or 1): {prediction[0]}")

    # Step 9: Simple Projection of Diabetes Risk Over 5 Years
    if prediction[0] == 1:
        # If the prediction is diabetes, assume a 5% increase in risk each year
        risk_over_years = [1]  # 1 means diabetes
    else:
        # If not diabetic, assume a potential risk based on factors
        risk_over_years = [0]  # 0 means no diabetes
        for year in range(1, 6):
            new_glucose = glucose * (1 + 0.05) ** year  # Simulating gradual increase in glucose
            new_bmi = bmi * (1 + 0.02) ** year  # Simulating gradual increase in BMI

            # Create a new DataFrame for the updated health parameters
            new_user_input = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [new_glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [new_bmi],
                'DiabetesPedigreeFunction': [dpf],
                'Age': [age + year]  # Age incremented by year
            })

            # Standardize new user input
            new_user_input_scaled = scaler.transform(new_user_input)
            future_prediction = best_model.predict(new_user_input_scaled)
            risk_over_years.append(future_prediction[0])

    # Output future diabetes risk over 5 years
    for i, risk in enumerate(risk_over_years):
        status = 'Diabetes' if risk == 1 else 'No Diabetes'
        print(f"Year {i}: {status} (Prediction value: {risk})")

# Step 10: Call the function to get user input and make predictions
predict_user_input_and_risk()

# Step 11: Plotting Predicted vs Actual values for Test Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='purple', edgecolor='k', alpha=0.6)
plt.title('Predicted vs Actual Values (Test Set)')
plt.xlabel('Actual Values (0 or 1)')
plt.ylabel('Predicted Values (0 or 1)')
plt.grid(True)
plt.show()