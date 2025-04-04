
![ML_Internship_PM](https://github.com/user-attachments/assets/76d53b9a-3fb6-4e55-81fd-a8dcb5492352)

<h1 align="center"> Turbofan Jet Engine RUL Prediction </h1>


<!-- This is a rgb line -->  
![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)


<!-- PROJECT DESCRIPTION -->
<h2> :pencil: Project Description </h2>

<p align="justify"> 
    Predictive maintenance is transforming aviation by leveraging machine learning to anticipate failures and optimize engine performance. This project focuses on accurately estimating the Remaining Useful Life (RUL) of aircraft turbofan engines using sensor data, enabling proactive maintenance to prevent costly downtime and catastrophic failures. By applying advanced regression and deep learning techniques, we analyse degradation patterns to enhance operational efficiency, reduce maintenance costs, and improve safety—critical for an industry where engine reliability impacts millions of dollars and lives. The solution not only extends engine longevity but also contributes to the economic sustainability of airlines in a highly competitive and regulated sector.  
</p>


![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)


<!-- PROJECT DOCUMENTATION -->
<h2> :bookmark: Project Documentation</h2>

<ul>
    <li><a href="https://github.com/KunalLokhande99/RULPrediction/blob/master/documents/01_Project_Architecture.pdf" target="_blank">Project Architecture</a></li>
    <li><a href="https://github.com/KunalLokhande99/RULPrediction/blob/master/documents/02_High_Level_Design_(HLD).pdf" target="_blank">High Level Design (HLD)</a></li>
    <li><a href="https://github.com/KunalLokhande99/RULPrediction/blob/master/documents/03_Low_Level_Design_(LLD).pdf" target="_blank">Low Level Design (LLD)</a></li>
    <li><a href="https://github.com/KunalLokhande99/RULPrediction/blob/master/documents/04_Project_Wireframe.pdf" target="_blank">Project Wireframe</a></li>
    <li><a href="https://github.com/KunalLokhande99/RULPrediction/blob/master/documents/05_Detailed_Project_Report_(DPR).pdf" target="_blank">Detailed Project Report</a></li>
</ul>


![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)


<!-- TECHNICAL SPECIFICATIONS -->
<h2> :hammer: Technical Specification</h2>

![TechnologyStackTable](https://github.com/user-attachments/assets/616fe3dd-d8a8-42b6-969c-c21c21f510f9)


![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)


<!-- DATASET DESCRIPTION -->
<h2> :floppy_disk: Dataset Description</h2>

**1. Source**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset.
    <ul>https://www.kaggle.com/datasets/behrad3d/nasa-cmaps</ul>
**2. Content**: Multivariate time-series data from turbofan jet engines, tracking 21 sensor readings (temperature, pressure, RPM) and 3 operational settings.  
**3. Scope**: Simulated run-to-failure cycles (20,631 records) under varying conditions (sea level, altitude, fault modes).  
**4. Preprocessing**: Normalized using Robust Scaling (median/IQR) to handle sensor outliers and noise.  
**5. Key Sensors**: Focus on critical degradation indicators (e.g., Sensor 7: exhaust temperature, Sensor 12: fuel flow).  
**6. Data Splits**:  
   - Training Data: 100 engines with full run-to-failure cycles (21 sensors × 20,631 cycles)  
   - Testing Data: 100 engines with partial cycles + final RUL values for validation  
   - RUL Labels: Provided separately for each engine in testing set 


![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)



<!-- RUL PREDICTION -->
<h2> :computer: RUL Prediction</h2>

<p align="justify"> 
    The project leverages machine learning models (Random Forest, Support Vector Machine, Gradient Boosting, XGBoost, etc) to analyze sensor degradation trends and predict RUL with 
cycle-level accuracy. These models process normalized telemetry data to generate maintenance alerts when RUL drops below operational thresholds, enabling proactive engine servicing. 
</p>

<h4> Perfomance Matrix </h4>

![Performance_Matrix](https://github.com/user-attachments/assets/6658dab3-5939-4b50-b9b7-6f0926d0d2ee)

<p></p> <!-- Empty line -->
<h4> Process Flow Diagram </h4>

![work_flow_Arc](https://github.com/user-attachments/assets/27f36e46-ea64-44f9-a876-b214271037c8)



![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)




<!-- MODEL DEPLOYMENT -->
<h2> :baggage_claim: Model Deployment</h2>

<p align="justify"> 
    The system is deployed as a local Flask web application running on `localhost:5001`, with a browser-based interface for real-time RUL predictions 
and maintenance alerts. All components (preprocessing logic, ML model, User Interface) are bundled into a single executable for easy offline use.   
</p>

<h4> User I/O Flow Diagram </h4>

![user_flow_Arc](https://github.com/user-attachments/assets/da5586c4-9b96-4e39-8861-9d3141199be3)

<p></p> <!-- Empty line -->
<h4> Web App Homepage </h4>

![Wireframe_Homepage](https://github.com/user-attachments/assets/a331c56c-bd1d-45ad-a61b-3a0971e0c84f)

![Wireframe_Result_Page](https://github.com/user-attachments/assets/c051eae6-87bd-4ece-9cf7-2610cbb34b97)



![rgb](https://github.com/user-attachments/assets/2f475ebb-3f56-4393-b921-9d70ff425996)




