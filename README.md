\# Mobile Phone Price Prediction Using Artificial Neural Network (ANN)



\## 📌 Project Overview



This project predicts the price of a mobile phone based on its technical specifications using an Artificial Neural Network (ANN).



The model takes 11 mobile phone features as input and predicts the estimated price in Indian Rupees (₹).



\## 🎯 Objective



The main objective of this project is to develop a machine learning model that can estimate mobile phone prices based on specifications such as RAM, storage, camera, battery capacity, processor cores, 5G support, and fast charging.



\## 📊 Features Used



The model uses the following 11 features:



1\. Battery Capacity (mAh)

2\. RAM (GB)

3\. Storage (GB)

4\. Screen Size (inches)

5\. Camera (MP)

6\. Front Camera (MP)

7\. Processor Cores

8\. Refresh Rate (Hz)

9\. Weight (g)

10\. 5G Support

11\. Fast Charging (W)



\### Target Variable



\- `price\_INR` — Mobile phone price in Indian Rupees



\## 🧠 ANN Architecture



The Artificial Neural Network contains the following layers:



\- Input Layer: 11 features

\- Dense Layer: 64 neurons, ReLU

\- Dense Layer: 32 neurons, ReLU

\- Dense Layer: 16 neurons, ReLU

\- Dense Layer: 8 neurons, ReLU

\- Output Layer: 1 neuron, Linear



The model is trained using the Adam optimizer and Mean Squared Error loss function.



\## 🔧 Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- TensorFlow

\- Keras

\- Tkinter

\- Git \& GitHub



\## ⚙️ Data Preprocessing



The dataset is divided into training and testing sets using a 70:30 split.



Feature scaling is performed using `StandardScaler`.



Early stopping is used during model training to help prevent unnecessary training.



\## 📈 Model Performance



The model achieved the following performance on the test dataset:



| Metric | Result |

|---|---:|

| MAE | ₹2,273.66 |

| MSE | 8,287,156.50 |

| RMSE | ₹2,878.74 |

| R² Score | 0.8965 |



\## 🖥️ GUI Application



A Tkinter-based graphical user interface is included in this project.



The user can enter mobile phone specifications and click the \*\*PREDICT PRICE\*\* button to get the estimated mobile phone price.



\## 📁 Project Structure



```text

Mobile\_Phone\_Price\_Prediction/

│

├── mobile\_phone\_price\_prediction.py

├── mobile\_price\_gui.py

├── mobile\_phone\_price\_prediction\_dataset.csv

├── mobile\_phone\_price\_model.keras

├── requirements.txt

└── README.md

