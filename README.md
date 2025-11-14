# Pepsi Stock
![Pepsi](/pepsi.jpg)

## Procedures
- Data Preparation and Acquistion
    - Data acquired from the yahoo finance api
- Feature Engineering
- Data Splitting
    - Split the data into training (80%) and testing (20%)
- Pre-Training Visualization

![pre-training-visualization](/output1.png)
- Feature Scaling
    - Initialize the StandardScaler (to normalize the data, which is cruical for linear models)
- Model Training and Comparison
    - Linear Regression
    - Ridge Regression (L2 Regularization)
    - Lasso Regression (L1 Regularization)
    - Random Forest Regressor
- Hyperparameter Tuning
- Post-Training Visualization (Best Performance)

![post-training-visualization](/output2.png)    
- Function for a New Prediction Input
# Final Model Comparison Summary

| Model                   | RMSE       | R² Score    |
|------------------------|------------|-------------|
| **Tuned Random Forest**     | **0.119478** | **0.999986** |
| Random Forest Regressor | 76.8756    | -4.70521    |
| Lasso Regression        | 3538.07    | -12083.5    |
| Ridge Regression        | 3651.53    | -12870.9    |
| Linear Regression       | 3652.32    | -12876.5    |

**Conclusion:**  
The best performing model based on RMSE is **Tuned Random Forest**.


## Process
![screenshot(254)](/Screenshot%20(254).png)
![screenshot(255)](/Screenshot%20(255).png)
![screenshot(256)](/Screenshot%20(256).png)
![screenshot(257)](/Screenshot%20(257).png)
![screenshot(258)](/Screenshot%20(258).png)

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/pepsi-stock.git
cd pepsi-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
pepsi-stock/
│
├── model.ipynb  
|── model.py    
|── pepsi_stock_data.csv  
├── requirements.txt 
├── pepsi.jpg   
|── output1.png
|── output2.png
├── Screenshot (254).png
├── Screenshot (255).png
├── Screenshot (256).png
├── Screenshot (257).png
├── Screenshot (258).png
|── LICENSE
└── README.md          

```
## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE)
 file for details.