# Interactive Sales Forecast Dashboard - Supermarket

This project uses **Streamlit** to create an interactive dashboard that performs revenue (sales) forecasts for a supermarket using a **Multi-Layer Perceptron (MLP)** model. The dashboard allows the visualization of historical sales data and makes predictions for the next three months.

## Features

- Displays historical sales data from a supermarket.
- Visualizes aggregated sales per month in a chart.
- Forecasts sales for the next three months using a Machine Learning model (MLP).
- Shows the **Mean Absolute Error (MAE)** of the prediction.

## Technologies Used

- **Python 3.x**
- **Streamlit**: Framework for creating interactive dashboards.
- **Pandas**: For data manipulation and aggregation.
- **Matplotlib**: To plot monthly sales charts.
- **Scikit-learn**: To train the MLP model and make predictions.

## Installation

### Prerequisites

Make sure you have **Python** installed in your environment. Additionally, you will need to install the dependencies listed in the `requirements.txt` file.

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repository
   ```

3. Create a virtual environment (optional, but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

If the `requirements.txt` file is not available, install the dependencies manually:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

## Running the Project

1. Make sure the CSV file containing the supermarket's sales data is in the same directory as the project, or adjust the path in the code as needed.
2. Run Streamlit:

   ```bash
   streamlit run branch_daily_dashboard.py
   ```

3. Streamlit will automatically open the default browser with the dashboard.

## Project Structure

```bash
.
├── .env
├── .data/
│   └── supermarket_sales.csv                # CSV file with historical sales data
├── notebooks/
│   └── branch_daily_regression_mlp.ipynb    # Main file containing the MLP code
├── scripts/
│   └── branch_daily_regression_mlp.py       # Main file containing the MLP code
├── branch_daily_dashboard.py                # Main file containing the dashboard code
├── LICENSE
├── README.md                                # This README file
└── requirements.txt                         # Project dependencies list
```

## Code Explanation

The main code is in the `branch_daily_dashboard.py` file and contains the following sections:

- **Data Loading**: Loads data from the CSV file and formats it for use in the dashboard.
- **Sales Visualization**: Generates a monthly sales chart by aggregating sales per month.
- **Model Training**: Trains an MLP (Multi-Layer Perceptron) model using monthly sales to forecast the next three months.
- **Sales Forecast**: Makes revenue predictions for the next three months and displays the results on the dashboard.

## Contributing

Feel free to contribute with improvements or new features! To contribute:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push the changes to the remote repository:
   ```bash
   git push origin new-feature
   ```
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for more details.

## Contact

- [@renanalencar](https://github.com/renanalencar)
