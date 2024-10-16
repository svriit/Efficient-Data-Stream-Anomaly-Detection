# Real-time Anomaly Detection with Adaptive Algorithms

## Overview

This project implements a real-time anomaly detection system designed to handle concept drift and seasonal variations in data streams. The system is built with an adaptive algorithm capable of effectively identifying anomalies while continuously adjusting to changes in data behavior. The project features a simulation of a data stream and a real-time visualization tool that displays both the data stream and detected anomalies.

## Project Features

- **Algorithm Selection**: An adaptive anomaly detection algorithm that accommodates concept drift and seasonal variations.
- **Data Stream Simulation**: Emulates real-time data flow with regular patterns, seasonal elements, and random noise.
- **Anomaly Detection**: A mechanism to detect anomalies in real-time as data is streamed.
- **Optimization**: Ensures speed and efficiency in anomaly detection.
- **Visualization**: Real-time visualization to display the data stream and highlight detected anomalies.

## Objectives

- **Algorithm Selection**: Identify and implement a suitable algorithm for anomaly detection, capable of adapting to concept drift and seasonal variations.
- **Data Stream Simulation**: Design a function to emulate a data stream, incorporating regular patterns, seasonal elements, and random noise.
- **Anomaly Detection**: Develop a real-time mechanism to accurately flag anomalies as the data is streamed.
- **Optimization**: Ensure the algorithm is optimized for both speed and efficiency.
- **Visualization**: Create a straightforward real-time visualization tool to display both the data stream and any detected anomalies.

## Prerequisites

Before running this application, ensure that you have Python installed (Python 3.7 or higher is recommended). You will also need to install the necessary Python packages listed below.

### Required Libraries

- numpy
- pandas
- scikit-learn
- matplotlib
- streamlit 
- plotly
- statsmodels

To install these packages, run the following command:

bash
pip install numpy pandas scikit-learn matplotlib streamlit plotly statsmodels

### How to Run the Application
- **Clone the Repository**

- clone the repository to your local machine using the following command:

bash
git clone <repository-url>

- Navigate to the project directory:

bash
cd real-time-anomaly-detection

- **Install Dependencies**

- Make sure you have all the required libraries installed. You can do this by running the command mentioned in the "Required Libraries" section.

**Run the Application**

- Start the Streamlit server to run the application by executing the following command:

bash
streamlit run app.py

- This command will open a web browser window showing the interface of the application.
-  If it doesn’t open automatically, you can navigate to *http://localhost:8501* to access the interface.

### User Interface Workflow

Step 1: Start Data Stream Simulation

- Click the Start Data Stream button to initiate the simulated data stream.
- The system will start generating data with regular patterns, seasonal elements, and random noise.

Step 2: Real-Time Anomaly Detection

- The adaptive anomaly detection algorithm continuously monitors the incoming data stream.
- Detected anomalies are flagged in real-time and highlighted on the visualization.

Step 3: **Visualization**

- The interface displays a real-time line chart of the data stream.
- Anomalies are marked as red dots on the chart for easy identification.

### Explanation of Code

**Imports and Setup**

- The application imports essential packages such as numpy for numerical calculations, pandas for data handling, scikit-learn for machine learning utilities, and streamlit for building the UI.
- A Streamlit title and buttons are defined to start and stop the data stream simulation.

### Data Stream Simulation

- A function simulates a real-time data stream. The generated data contains seasonal patterns, regular trends, and random noise to emulate real-world scenarios.
- numpy is used to create sinusoidal data representing seasonal effects, combined with linear trends and Gaussian noise.

### Adaptive Anomaly Detection Algorithm

- The chosen anomaly detection algorithm can adapt to concept drift and seasonal variations. Techniques like the Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD) or Online ARIMA are suitable for this use case.
- The model is continuously updated as new data arrives to accommodate changes in the data pattern, helping prevent stale models.

### Real-Time Detection and Visualization

- Anomalies are flagged based on the deviation from expected seasonal and trend behavior.
- A Plotly line chart is used for real-time visualization. The data stream is plotted with anomalies highlighted using red markers.

The visualization is updated dynamically as new data points are added to the stream.

### Optimization

- The system is optimized to handle high-frequency data streams efficiently.
- Techniques like batch processing of data points and incremental learning are used to maintain a balance between speed and accuracy.

### Important Notes

**Real-Time Limitations**: The system is optimized for simulated real-time data. In a real-world scenario, data latency might affect detection accuracy.

**Algorithm Adaptability**: The chosen algorithm should be capable of handling concept drift and seasonal variations, which makes it ideal for continuously evolving datasets.

**Internet Connection**: The application does not require an internet connection once the dependencies are installed, as all processing is done locally.

### Troubleshooting

- If the application does not start, make sure all the required dependencies are installed and that you are using a compatible Python version.
- If anomalies are not being detected correctly, consider adjusting the parameters of the detection algorithm (e.g., sensitivity thresholds).

