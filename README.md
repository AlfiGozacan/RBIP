# RBIP
The aim of this project is to build a data-driven Risk-Based Inspection Programme (RBIP).

The file combined_code.py contains the Python code that cleans and imputes the data, while training the ML models. It then produces output in the form of risk quartiles for every UPRN in the RBIP database, along with a few diagnostics for the models. The file address_scores_join.py just adds on the addresses to the csv output. Also contained in output.csv is all the values of the features the model used to predict risk.
