#imports

##Libs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##Data
pag_hei = pd.read_csv("C:/Users/vitoria.novello_ifoo/Documents/tera/Inferencia-estatistica_saude-mental/data/PAG_HEI.csv")
demo_phq = pd.read_csv("C:/Users/vitoria.novello_ifoo/Documents/tera/Inferencia-estatistica_saude-mental/data/DEMO_PHQ.csv")

#Handling Data

##Join das bases
df = demo_phq.merge(pag_hei, on = 'SEQN', how = 'left')
df.count()

##Substituindo 7 e 9 por valores ausentes nas colunas DPQ 
df2 = df
df2.iloc[:, 1:10] = df2.iloc[:, 1:10].replace({7:np.nan, 9:np.nan})

replace_map = {
  "RIDRETH1": {5: 2}, # Other
  "DMDEDUC": {7: np.nan, 9: np.nan},
  "INDFMINC": {1: np.mean([0,4999]), 2: np.mean([5000,9999]),
               3: np.mean([10000,14999]),4: np.mean([15000,19999]),
               5: np.mean([20000,24999]),6: np.mean([25000,34999]),
               7: np.mean([35000,44999]), 8: np.mean([45000,54999]),
               9: np.mean([55000,64999]), 10: np.mean([65000,74999]),
               11: 75000, 12: np.mean([20000, 90000]), 13: np.mean([0, 19999]),
               77: np.nan, 99: np.nan}
}

df2 = df2.replace(replace_map)

##Analise valores ausentes
df2.isna().sum().sort_values(ascending=False)
df2.shape[0]

100*df2.isna().sum().sort_values(ascending=False)/df2.shape[0]

##Criando nova variavel PHQ

df2["phq9"] = df2.iloc[:, 1:10].sum(axis = 'columns', skipna = False)
df2['phq9'].describe()

##Criação da variavel ph1_grp

conditions = [
  (df2['phq9'].isna()),
  (df2['phq9'] < 5),
  (df2['phq9'] >= 5) & (df2['phq9'] <= 9),
  (df2['phq9'] > 9) & (df2['phq9'] <= 14),
  (df2['phq9'] > 14) & (df2['phq9'] <= 19),
  (df2['phq9'] > 19)
    ]
values = [np.nan, 0, 1, 2, 3, 4]
df2["phq_grp"] = np.select(conditions, values) # Construindo a variável
df2[["phq_grp"]].value_counts(sort = False) # Avaliando as frequências

##Agrupamento de categorias 2,3,4 

df2["phq_grp2"] = df2["phq_grp"].replace([3, 4], 2)
df2[["phq_grp2"]].value_counts(sort = False)

#Definindo as categorias de acordo com o dicionario das bases
label_quali = {
  "RIAGENDR": {1: 'Masculino', 2: 'Feminino'},
  "RIDRETH1": {1: 'Americano Mexicano', 2: 'Outro',
               3: 'Branco \n não hispânico', 4: 'Negro \n não hispânico'},
  "DMDEDUC": {1: "< 9 ano", 2: "9-12 ano", 3: "Ensino \n médio",
              4: "Superior \n incompleto", 5: "Superior \n completo"},
  "ADHERENCE": {1: 'Baixo', 2: 'Adequado', 3: 'Acima'},
  "phq_grp2": {0: "Sem sintomas", 1: "Sintomas \n leves",
               2: "Sintomas \n moderados-severos"}
}

df2 = df2.replace(label_quali)

#Separação das variaveis entre quantitativas e qualitativas
var_quant = [
    "RIDAGEYR", 
    "INDFMINC", 
    "PAG_MINW", 
    "HEI2015C1_TOTALVEG",
    "HEI2015C2_GREEN_AND_BEAN",
    "HEI2015C3_TOTALFRUIT",
    "HEI2015C4_WHOLEFRUIT",
    "HEI2015C5_WHOLEGRAIN",
    "HEI2015C6_TOTALDAIRY",
    "HEI2015C7_TOTPROT",
    "HEI2015C8_SEAPLANT_PROT",
    "HEI2015C9_FATTYACID",
    "HEI2015C10_SODIUM",
    "HEI2015C11_REFINEDGRAIN",
    "HEI2015C12_SFAT",
    "HEI2015C13_ADDSUG",
    "HEI2015_TOTAL_SCORE",
    "phq9"]

var_quali = [
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC",
    "ADHERENCE",
    "phq_grp2"
]

