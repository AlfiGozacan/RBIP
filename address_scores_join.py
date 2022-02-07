import pandas as pd
import pyodbc

server = "HQCFRMISSQL"
database = "CFRMIS_HUMBS"

cnxn = pyodbc.connect("DRIVER={SQL Server};SERVER="+server+";DATABASE="+database)

query = '''
select *
from ADDRESS_GAZ
where PREMISE_DESCRIPTION like 'Commercial%'
'''

addresses = pd.read_sql(query, cnxn)

addresses.drop_duplicates(subset="UPRN", keep="first", inplace=True)

file_path = "C:\\path_to_data\\"

scores = pd.read_csv(file_path+"output.csv")

df = addresses.merge(right=scores, left_on="UPRN", right_on="UPRN", how="inner")

df.to_csv(file_path+"output_with_addresses.csv", index=False)