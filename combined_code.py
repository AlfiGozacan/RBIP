### Load libraries
print("Loading libraries...")

import numpy as np
import pandas as pd
import pyodbc
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer

### Clean data
print("Cleaning data...")

file_path = "C:\\Users\\agozacan\\OneDrive - Humberside Fire and Rescue Service\\RBIP Project\\Input and Output\\"

loc_auths = ["ERY", "Hull", "NLincs", "NELincs"]

for i in range(len(loc_auths)):

    if i == 0:

        epc_df = pd.read_csv(file_path + "certificates_" + loc_auths[i] + ".csv")

    else:

        epc_df = pd.concat([epc_df, pd.read_csv(file_path +
                                                "certificates_" +
                                                loc_auths[i] +
                                                ".csv")]).reset_index(drop=True)

epc_df.drop(epc_df[epc_df["ASSET_RATING_BAND"].isin(["A+", "INVALID!"])].index,
            axis=0,
            inplace=True)

epc_df.reset_index(drop=True, inplace=True)

epc_df.replace(np.nan, "", inplace=True)

server = "HQCFRMISSQL"

database = "CFRMIS_HUMBS"

cnxn = pyodbc.connect("DRIVER={SQL Server};SERVER="+server+";DATABASE="+database)

query = '''
select *
from ADDRESS_GAZ
where PREMISE_DESCRIPTION like 'Commercial%'
'''

address_df = pd.read_sql(query, cnxn)

address_strings = []

for i in range(len(address_df)):

    string = " ".join(entry for entry in address_df.iloc[i, 5:10])

    address_strings.append(string)

epc_strings = []

for i in epc_df[epc_df["UPRN"] == ""].index:

    string = " ".join(entry for entry in epc_df.iloc[i, 2:5])

    epc_strings.append(string)

matching_indices = []

for i in tqdm(range(len(epc_strings))):

    fuzz_ratios = []

    for j in range(len(address_strings)):

        if epc_strings[i][-8:] == address_strings[j][-8:]:

            fuzz_ratios.append(fuzz.token_set_ratio(epc_strings[i], address_strings[j]))

        else:

            fuzz_ratios.append(0)

    index = fuzz_ratios.index(max(fuzz_ratios))

    matching_indices.append(index)

epc_df.loc[epc_df[epc_df["UPRN"] == ""].index, "UPRN"] = list(address_df.iloc[matching_indices, 1])

duplicate_UPRNs = epc_df["UPRN"].value_counts().rename_axis("UPRN").reset_index(name="COUNT")

duplicate_UPRNs.drop(duplicate_UPRNs.index[duplicate_UPRNs["COUNT"] == 1], axis=0, inplace=True)

print("\n")

for i in tqdm(range(len(duplicate_UPRNs))):

    sub_df = epc_df[epc_df["UPRN"] == duplicate_UPRNs.iloc[i,0]]

    new_entry = sub_df.iloc[0,:].copy()

    new_entry["ASSET_RATING"] = np.mean(sub_df["ASSET_RATING"])

    new_entry["ASSET_RATING_BAND"] = stats.mode(sub_df["ASSET_RATING_BAND"])[0][0]

    epc_df.drop(sub_df.index, axis=0, inplace=True)

    epc_df.loc[sub_df.index[0]] = new_entry

epc_df.reset_index(drop=True, inplace=True)

query = '''
select UPRN,
       CUSTODIAN,
       RISK_OF_FIRE,
       SEVERITY_OF_FIRE,
       SLEEPING_RISK,
       SLEEPING_RISK_ABOVE,
       SLEEPING_RISK_SCORE,
       SSRI_SCORE
from RBIP_2021
where RISK_OF_FIRE != 'NULL'
and SEVERITY_OF_FIRE != 'NULL'
and SLEEPING_RISK != 'NULL'
and SLEEPING_RISK_ABOVE != 'NULL'
and SLEEPING_RISK_SCORE != 'NULL'
and SSRI_SCORE != 'NULL'
'''

rbip_df = pd.read_sql(query, cnxn)

epc_df["UPRN"] = [int(uprn) for uprn in epc_df["UPRN"]]

rbip_df["UPRN"] = [int(uprn) for uprn in rbip_df["UPRN"]]

rbip_epc_df = rbip_df.merge(right=epc_df, left_on="UPRN", right_on="UPRN", how="outer")

server = "HQIRS"

database = "threetc_irs"

cnxn = pyodbc.connect("DRIVER={SQL Server};SERVER="+server+";DATABASE="+database)

query = '''
select  V_VISION_INCIDENT.REVISED_INCIDENT_TYPE,
        V_VISION_INCIDENT.GAZETTEER_URN,
        V_VISION_INCIDENT.CREATION_DATE,
        inc_incident.inc_location_address,
        inc_incident.inc_property_type
from V_VISION_INCIDENT
join inc_incident
on V_VISION_INCIDENT.INCIDENT_NUMBER = inc_incident.inc_incident_ref
where isnumeric(GAZETTEER_URN) = 1
and inc_incident.inc_property_type in (
    select CODE
    from Reporting_MENU_IRS_PROPERTY_TYPE
    where CATEGORY = 'Commercial'
)
'''

inc_df = pd.read_sql(query, cnxn)

inc_df["GAZETTEER_URN"] = [int(uprn) for uprn in inc_df["GAZETTEER_URN"]]

inc_df["YEAR"] = ["inc." + str(inc_df.loc[i, "CREATION_DATE"])[:4] for i in range(len(inc_df))]

inc_crosstab = pd.crosstab(inc_df["GAZETTEER_URN"],
                           inc_df["YEAR"]).rename_axis("UPRN").reset_index()

rbip_epc_inc_df = rbip_epc_df.merge(right=inc_crosstab, left_on="UPRN", right_on="UPRN", how="left")

model_cols = ["UPRN",
              "CUSTODIAN",
              "RISK_OF_FIRE",
              "SEVERITY_OF_FIRE",
              "SLEEPING_RISK",
              "SLEEPING_RISK_ABOVE",
              "SSRI_SCORE",
              "ASSET_RATING",
              "ASSET_RATING_BAND",
              "PROPERTY_TYPE",
              "MAIN_HEATING_FUEL",
              "FLOOR_AREA",
              "BUILDING_EMISSIONS",
              "PRIMARY_ENERGY_VALUE",
              "inc.2010",
              "inc.2011",
              "inc.2012",
              "inc.2013",
              "inc.2014",
              "inc.2015",
              "inc.2016",
              "inc.2017",
              "inc.2018",
              "inc.2019",
              "inc.2020"]

rbip_epc_inc_df["ASSET_RATING_BAND"] = rbip_epc_inc_df["ASSET_RATING_BAND"].replace(["A",
                                                           "B",
                                                           "C",
                                                           "D",
                                                           "E",
                                                           "F",
                                                           "G"], range(7))

rbip_epc_inc_df["CUSTODIAN"] = rbip_epc_inc_df["CUSTODIAN"].replace(
                                                    ["Crew",
                                                     "Protection Team"],
                                                    [0, 1])

categorical_cols = ["SLEEPING_RISK",
                    "PROPERTY_TYPE",
                    "MAIN_HEATING_FUEL"]

numerical_cols = ["RISK_OF_FIRE",
                  "SEVERITY_OF_FIRE",
                  "SLEEPING_RISK_ABOVE",
                  "SSRI_SCORE",
                  "ASSET_RATING",
                  "ASSET_RATING_BAND",
                  "FLOOR_AREA",
                  "BUILDING_EMISSIONS",
                  "PRIMARY_ENERGY_VALUE"]

rbip_epc_inc_df = rbip_epc_inc_df[model_cols]

rbip_epc_inc_df.replace("", 0, inplace=True)

model_df = rbip_epc_inc_df.copy()

model_df.dropna(axis=0, subset=numerical_cols, inplace=True)

model_df.replace(np.nan, 0, inplace=True)

model_df.reset_index(drop=True, inplace=True)

impute_df = rbip_epc_inc_df.copy()

for j in range(len(categorical_cols)):

    null_indices = impute_df.index[impute_df[categorical_cols[j]].isna()]

    non_null_indices = impute_df.index[~impute_df[categorical_cols[j]].isna()]

    mode = stats.mode(impute_df.loc[non_null_indices, categorical_cols[j]])[0][0]

    impute_df.loc[null_indices, categorical_cols[j]] = mode

null_indices = impute_df.index[impute_df["CUSTODIAN"].isna()]

impute_df.loc[null_indices, "CUSTODIAN"] = 0

imputer = KNNImputer(weights="distance")

impute_df.loc[:,numerical_cols] = imputer.fit_transform(impute_df[numerical_cols])

impute_df.replace(np.nan, 0, inplace=True)

model_df.loc[model_df[model_df["inc.2020"] > 0].index, "inc.2020"] = 1

model_df.rename({"inc.2020": "inc.2020.bool"}, axis=1, inplace=True)

encoder = OneHotEncoder(drop="first", sparse=False)

dummy_view = encoder.fit_transform(model_df[categorical_cols])

encoded_model_df = pd.DataFrame(dummy_view)

encoded_model_df.columns = encoder.get_feature_names(categorical_cols)

model_df.drop(categorical_cols, axis=1, inplace=True)

model_df = encoded_model_df.join(model_df)

cols = model_df.columns.tolist()

cols.remove("UPRN")

cols.remove("CUSTODIAN")

cols.insert(0, "CUSTODIAN")

cols.insert(0, "UPRN")

model_df = model_df[cols]

impute_df.loc[impute_df[impute_df["inc.2020"] > 0].index, "inc.2020"] = 1

impute_df.rename({"inc.2020": "inc.2020.bool"}, axis=1, inplace=True)

encoder = OneHotEncoder(drop="first", sparse=False)

dummy_view = encoder.fit_transform(impute_df[categorical_cols])

encoded_impute_df = pd.DataFrame(dummy_view)

encoded_impute_df.columns = encoder.get_feature_names(categorical_cols)

impute_df.drop(categorical_cols, axis=1, inplace=True)

impute_df = encoded_impute_df.join(impute_df)

cols = impute_df.columns.tolist()

cols.remove("UPRN")

cols.remove("CUSTODIAN")

cols.insert(0, "CUSTODIAN")

cols.insert(0, "UPRN")

impute_df = impute_df[cols]

surplus_columns = [col for col in impute_df.columns if col not in model_df.columns]

impute_df.drop(surplus_columns, axis=1, inplace=True)

ncols = len(model_df.columns)

### Train model
print("Training model...")

training_set, test_set = train_test_split(model_df, test_size = 0.33, random_state=1)

oversamp = RandomOverSampler(random_state=1)

X, y = oversamp.fit_resample(training_set.iloc[:,:-1], training_set.iloc[:,-1])

training_set = pd.DataFrame(X)

training_set["inc.2020.bool"] = y

X_train = training_set.iloc[:,2:-1]
y_train = training_set.iloc[:,-1]
X_test = test_set.iloc[:,2:-1]
y_test = test_set.iloc[:,-1]

adaboost = AdaBoostClassifier(random_state=1)
adaboost.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)

logreg = LogisticRegression(random_state=1, solver="liblinear")
logreg.fit(X_train, y_train)

xgboost = GradientBoostingClassifier(random_state=1)
xgboost.fit(X_train, y_train)

mlp = MLPClassifier(random_state=1)
mlp.fit(X_train, y_train)

### Evaluate model
print("Evaluating model...")

y_ada_pred = adaboost.predict(X_test)
test_set.insert(ncols, "AdaBoost Predictions", y_ada_pred)

y_rf_pred = (rf.predict_proba(X_test)[:,1] >= 0.3).astype(float)
test_set.insert(ncols+1, "RF Predictions", y_rf_pred)

y_lr_pred = logreg.predict(X_test)
test_set.insert(ncols+2, "LogReg Predictions", y_lr_pred)

y_xg_pred = xgboost.predict(X_test)
test_set.insert(ncols+3, "XGBoost Predictions", y_xg_pred)

y_mlp_pred = mlp.predict(X_test)
test_set.insert(ncols+4, "MLP Predictions", y_mlp_pred)

real_positives = len(test_set[test_set["inc.2020.bool"] == 1.0])
adaboost_positives = len(test_set[test_set["AdaBoost Predictions"] == 1.0])
rf_positives = len(test_set[test_set["RF Predictions"] == 1.0])
logreg_positives = len(test_set[test_set["LogReg Predictions"] == 1.0])
XGBoost_positives = len(test_set[test_set["XGBoost Predictions"] == 1.0])
MLP_positives = len(test_set[test_set["MLP Predictions"] == 1.0])

print(f'''There are {len(test_set)} entries in the test set,
       of which {real_positives} are real positives''')
print(f"AdaBoost predicted {adaboost_positives} positives")
print(f"Random Forest predicted {rf_positives} positives")
print(f"Logistic Regression predicted {logreg_positives} positives")
print(f"XGBoost predicted {XGBoost_positives} positives")
print(f"MLP predicted {MLP_positives} positives")

print("AdaBoost:\n", classification_report(test_set.iloc[:,ncols-1],
                                           test_set.iloc[:,ncols]))
print("Random Forest:\n", classification_report(test_set.iloc[:,ncols-1],
                                                test_set.iloc[:,ncols+1]))
print("Logistic Regression:\n", classification_report(test_set.iloc[:,ncols-1],
                                                      test_set.iloc[:,ncols+2]))
print("XGBoost:\n", classification_report(test_set.iloc[:,ncols-1],
                                          test_set.iloc[:,ncols+3]))
print("MLP:\n", classification_report(test_set.iloc[:,ncols-1],
                                      test_set.iloc[:,ncols+4]))

length = len(test_set.iloc[:,ncols-1])

ada_no_matched = sum([(test_set.iloc[i,ncols-1] * test_set.iloc[i,ncols]) +
                      ((1-test_set.iloc[i,ncols-1]) * (1-test_set.iloc[i,ncols]))
                      for i in range(length)])
rf_no_matched = sum([(test_set.iloc[i,ncols-1] * test_set.iloc[i,ncols+1]) +
                     ((1-test_set.iloc[i,ncols-1]) * (1-test_set.iloc[i,ncols+1]))
                     for i in range(length)])
lr_no_matched = sum([(test_set.iloc[i,ncols-1] * test_set.iloc[i,ncols+2]) +
                     ((1-test_set.iloc[i,ncols-1]) * (1-test_set.iloc[i,ncols+2]))
                     for i in range(length)])
xg_no_matched = sum([(test_set.iloc[i,ncols-1] * test_set.iloc[i,ncols+3]) +
                     ((1-test_set.iloc[i,ncols-1]) * (1-test_set.iloc[i,ncols+3]))
                     for i in range(length)])
mlp_no_matched = sum([(test_set.iloc[i,ncols-1] * test_set.iloc[i,ncols+4]) +
                      ((1-test_set.iloc[i,ncols-1]) * (1-test_set.iloc[i,ncols+4]))
                      for i in range(length)])

ada_accuracy = ada_no_matched / length
rf_accuracy = rf_no_matched / length
lr_accuracy = lr_no_matched / length
xg_accuracy = xg_no_matched / length
mlp_accuracy = mlp_no_matched / length

print("AdaBoost Proportion Correctly Guessed:", ada_accuracy)
print("Random Forest Proportion Correctly Guessed:", rf_accuracy)
print("Logistic Regression Proportion Correctly Guessed:", lr_accuracy)
print("XGBoost Proportion Correctly Guessed:", xg_accuracy)
print("MLP Proportion Correctly Guessed:", mlp_accuracy)

### PRODUCE OUTPUT PLOTS / TABLES
print("Producing output...")

all_adaprobs = adaboost.predict_proba(impute_df.iloc[:,2:-1])
adaprobs = adaboost.predict_proba(X_test)
rfprobs = rf.predict_proba(X_test)
lrprobs = logreg.predict_proba(X_test)
xgprobs = xgboost.predict_proba(X_test)
mlpprobs = mlp.predict_proba(X_test)

positive_probs = (np.array([x[1] for x in all_adaprobs]) +
                  np.random.random(size=len(impute_df)) * 1e-5)

complex_indices = impute_df.index[impute_df["CUSTODIAN"] == 1]

non_complex_indices = impute_df.index[impute_df["CUSTODIAN"] == 0]

complex_positive_probs = [positive_probs[i] for i in complex_indices]

non_complex_positive_probs = [positive_probs[i] for i in non_complex_indices]

impute_df.loc[complex_indices, "QUARTILE"] = pd.qcut(complex_positive_probs,
                                                     q=4,
                                                     labels=[4, 3, 2, 1])

impute_df.loc[non_complex_indices, "QUARTILE"] = pd.qcut(non_complex_positive_probs,
                                                         q=4,
                                                         labels=[4, 3, 2, 1])

impute_df.to_csv(file_path + "output.csv", index=False)

probas = [adaprobs, rfprobs, lrprobs, xgprobs, mlpprobs]
titles = ["AdaBoost", "Random Forest", "Logistic Regression", "XGBoost", "MLP"]

for i in range(len(probas)):
    
    skplt.metrics.plot_roc(y_test, probas[i], title=titles[i])

    plt.savefig(file_path+"roc_"+str(i)+".png", dpi = 200, bbox_inches = "tight")

features = rf.feature_importances_

ftrs = pd.DataFrame({"column_name": model_df.columns[2:-1],
                     "score": features}).sort_values(
                                        by="score",
                                        ascending=False).reset_index(drop=True)

ftrs.to_csv(file_path + "feature_importance_scores.csv", index=False)

plt.figure(figsize=(10,8))
sns.barplot(y = ftrs.loc[:20, "column_name"], x = ftrs.loc[:20, "score"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Score")
plt.ylabel("Feature Name")
plt.savefig(file_path+"feature_importance.png", dpi = 200, bbox_inches = "tight")

### Complete
print("Done.")