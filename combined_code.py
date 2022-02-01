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
from sklearn.metrics import classification_report

### Clean data
print("Cleaning data...")

file_path = "C:\\path_to_data\\"

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

rbip_epc_df = rbip_df.merge(right=epc_df, left_on="UPRN", right_on="UPRN", how="inner")

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

df = rbip_epc_df.merge(right=inc_crosstab, left_on="UPRN", right_on="UPRN", how="inner")

model_cols = ["UPRN",
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

df["ASSET_RATING_BAND"] = df["ASSET_RATING_BAND"].replace(["A",
                                                           "B",
                                                           "C",
                                                           "D",
                                                           "E",
                                                           "F",
                                                           "G"], range(7))

categorical_cols = ["SLEEPING_RISK",
                    "PROPERTY_TYPE",
                    "MAIN_HEATING_FUEL"]

df = df[model_cols]

df.replace(np.nan, 0, inplace=True)
df.replace("", 0, inplace=True)

df.loc[df[df["inc.2020"] > 0].index, "inc.2020"] = 1

df.rename({"inc.2020": "inc.2020.bool"}, axis=1, inplace=True)

encoder = OneHotEncoder(drop="first", sparse=False)

dummy_view = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(dummy_view)

encoded_df.columns = encoder.get_feature_names(categorical_cols)

df.drop(categorical_cols, axis=1, inplace=True)

df = encoded_df.join(df)

cols = df.columns.tolist()
cols.remove("UPRN")
cols.insert(0, "UPRN")
df = df[cols]

ncols = len(df.columns)

### Train model
print("Training model...")

training_set, test_set = train_test_split(df, test_size = 0.33, random_state=1)

oversamp = RandomOverSampler(random_state=1)

X, y = oversamp.fit_resample(training_set.iloc[:,:-1], training_set.iloc[:,-1])

training_set = pd.DataFrame(X)

training_set["inc.2020.bool"] = y

X_train = training_set.iloc[:,1:-1]
y_train = training_set.iloc[:,-1]
X_test = test_set.iloc[:,1:-1]
y_test = test_set.iloc[:,-1]

adaboost = AdaBoostClassifier(random_state=1)
adaboost.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)

logreg = LogisticRegression(random_state=1, solver="liblinear")
logreg.fit(X_train, y_train)

xgboost = GradientBoostingClassifier(random_state=1)
xgboost.fit(X_train, y_train)

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

real_positives = len(test_set[test_set["inc.2020.bool"] == 1.0])
adaboost_positives = len(test_set[test_set["AdaBoost Predictions"] == 1.0])
rf_positives = len(test_set[test_set["RF Predictions"] == 1.0])
logreg_positives = len(test_set[test_set["LogReg Predictions"] == 1.0])
XGBoost_positives = len(test_set[test_set["XGBoost Predictions"] == 1.0])

print(f'''There are {len(test_set)} entries in the test set,
       of which {real_positives} are real positives''')
print(f"AdaBoost predicted {adaboost_positives} positives")
print(f"Random Forest predicted {rf_positives} positives")
print(f"Logistic Regression predicted {logreg_positives} positives")
print(f"XGBoost predicted {XGBoost_positives} positives")

print("AdaBoost:\n", classification_report(test_set.iloc[:,ncols-1],
                                           test_set.iloc[:,ncols]))
print("Random Forest:\n", classification_report(test_set.iloc[:,ncols-1],
                                                test_set.iloc[:,ncols+1]))
print("Logistic Regression:\n", classification_report(test_set.iloc[:,ncols-1],
                                                      test_set.iloc[:,ncols+2]))
print("XGBoost:\n", classification_report(test_set.iloc[:,ncols-1],
                                          test_set.iloc[:,ncols+3]))

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

ada_accuracy = ada_no_matched / length
rf_accuracy = rf_no_matched / length
lr_accuracy = lr_no_matched / length
xg_accuracy = xg_no_matched / length

print("AdaBoost Proportion Correctly Guessed:", ada_accuracy)
print("Random Forest Proportion Correctly Guessed:", rf_accuracy)
print("Logistic Regression Proportion Correctly Guessed:", lr_accuracy)
print("XGBoost Proportion Correctly Guessed:", xg_accuracy)

all_adaprobs = adaboost.predict_proba(df.iloc[:,1:-1])
adaprobs = adaboost.predict_proba(X_test)
rfprobs = rf.predict_proba(X_test)
lrprobs = logreg.predict_proba(X_test)
xgprobs = xgboost.predict_proba(X_test)

positive_probs = [x[1] for x in all_adaprobs]

df.loc[:, "QUARTILE"] = pd.qcut(positive_probs, q=4, labels=[4, 3, 2, 1])

df.to_csv(file_path + "output.csv", index=False)

probas = [adaprobs, rfprobs, lrprobs, xgprobs]
titles = ["AdaBoost", "Random Forest", "Logistic Regression", "XGBoost"]

for i in range(len(probas)):
    
    skplt.metrics.plot_roc(y_test, probas[i], title=titles[i])

plt.show()

features = rf.feature_importances_

ftrs = pd.DataFrame({"column_name": df.columns[1:-2],
                     "score": features}).sort_values(
                                        by="score",
                                        ascending=False).reset_index(drop=True)

ftrs.to_csv(file_path + "feature_importance_scores.csv", index=False)

plt.figure(figsize=(10,8))
sns.barplot(y = ftrs.loc[:30, "column_name"], x = ftrs.loc[:30, "score"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Score")
plt.ylabel("Feature Names")
plt.show()