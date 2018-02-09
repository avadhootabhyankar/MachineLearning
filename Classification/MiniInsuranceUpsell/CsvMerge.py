import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
MiniHokenMoshikomi_Anketo_RIT_Report = pd.read_csv('MiniHokenMoshikomi_Anketo_RIT_Report.csv')
print(MiniHokenMoshikomi_Anketo_RIT_Report.head())

MiniInsuranceUpsell201802 = pd.read_csv('MiniInsuranceUpsell201802.csv')
print(MiniInsuranceUpsell201802.head())

result1 = pd.merge(MiniHokenMoshikomi_Anketo_RIT_Report, MiniInsuranceUpsell201802, on='楽天会員ID', how='left')
print(result1.head())
#df.merge(df1, left_index=True, right_index=True, how='left')

Opportunity_Product_RIT_Report = pd.read_csv('Opportunity_Product_RIT_Report.csv')
result2 = pd.merge(result1, Opportunity_Product_RIT_Report, on='楽天会員ID', how='left')
print(result2.head())

result3 = result2["ANP"]
result3.loc[~result3.isnull()] = 1  # not nan
result3.loc[result3.isnull()] = 0   # nan

print(result3.unique())

result4 = result2[['楽天会員ID','受付日','加入ステータス','性別_x','生年月日','回答1','回答2','回答2-1','回答2-2']]
#result5 = pd.concat([result4, result3], ignore_index=True) 
result5 = pd.concat([result4, result3], axis=1)
result5.rename(columns={'楽天会員ID': 'EasyId', '加入ステータス':'EntryStatus', '生年月日':'Birthdate', '回答1':'A1', '回答2':'A2', '回答2-1':'A2-1', '回答2-2':'A2-2'}, inplace=True)
result5.rename(columns={'ANP': 'IsUpselled'}, inplace=True)
result5.rename(columns={'性別_x': 'Gender'}, inplace=True)
result5 = result5.dropna()
result5['ApplyYear'] = result5.受付日.str.slice(0, 4)
result5['ApplyMonth'] = result5.受付日.str.slice(5, 6)
result5 = result5.drop(['受付日'], axis = 1)

result5['Birthdate'] = pd.to_datetime(result5['Birthdate'])
now = dt.datetime.now()
result5['Age'] = ((now - result5['Birthdate'])).astype(str)
#print(result5.describe())
result5['Age'] = result5.Age.str.slice(0, 5)
result5['Age'] = result5['Age'].astype(int) // 365
result5 = result5.drop(['Birthdate'], axis = 1)

print("final result : \n", result5.head())
result5.to_csv('IsUpselledMini.csv', index=False)
