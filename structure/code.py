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

    
df2['PAG_MINW_trunc'] = np.where(df2['PAG_MINW'] > 3600, 3600, df2['PAG_MINW'])
df2[['PAG_MINW', 'PAG_MINW_trunc']].describe(percentiles = [.01, .25, .5, .75, .99]).round(2)

# Criando a variável PAG_MIN em horas
df2['PAG_HW'] = df2['PAG_MINW_trunc']/60

var_quant = [
    "RIDAGEYR", 
    "INDFMINC", 
    "PAG_HW", 
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

for var in var_quant:
  sns.histplot(df2[var], kde=True)
  plt.show()

for var in var_quali:
   print(df2[var].value_counts())

for var in var_quali:
    print(100*df2[var].value_counts(normalize = True))

def grafico_barras_prop(data, variable):
    (data[[variable]]
     .value_counts(normalize = True, sort = False)
     .rename("Proporção")
     .reset_index()
     .pipe((sns.barplot, "data"), x=variable, y="Proporção"))
    plt.ylim(0,1)
    plt.show()

for var in var_quali:
    grafico_barras_prop(df2, var)
    plt.show()

# Atualizando a lista de qualitativas para excluir phq_grp

var_quali = ["RIAGENDR", "RIDRETH1", "DMDEDUC", "ADHERENCE"]

var_quanti = ["RIDAGEYR", "INDFMINC",
              "PAG_HW", "HEI2015C1_TOTALVEG",
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
              "HEI2015_TOTAL_SCORE"]

from IPython.display import display

for var in var_quant:
    display(df2[['phq_grp2', var]].groupby('phq_grp2').describe().round(2))

def grafico_boxplot_grp(data, variable, label):
    
    if label == "": label = variable
    sns.boxplot(x="phq_grp2", y=variable, data=data.replace({'phq_grp2': {0: "Sem sintomas", 
                                                                          1: "Sintomas leves",
                                                                          2: "Sintomas mod-graves"}}))
    plt.ylabel(label)
    plt.show()

def grafico_barras_prop_grp(data, variable):
    (data
     .groupby(variable)['phq_grp2']
     .value_counts(normalize = True, sort = False)
     .rename("Proporção")
     .reset_index()
     .pipe((sns.catplot, "data"), x=variable, y="Proporção", hue = 'phq_grp2', kind="bar"))
    plt.ylim(0,1)
    plt.show()

for var in var_quali:
    grafico_barras_prop_grp(df2, var)

#Testes de hipótese
##Há associação entre gênero e depressão?

grafico_barras_prop_grp(df2, 'RIAGENDR')

##Qui-quadrado de independência - A ideia é comparar as proporções de sintomas de depressão entre gênero

100*pd.crosstab(df2['RIAGENDR'],
                df2['phq_grp2'],
                normalize='index')

crosstab = pd.crosstab(df2['RIAGENDR'],
                       df2['phq_grp2'])
crosstab

# O valor de p do teste foi p < 0,001. A hipótese foi rejeitada, e portanto há associação entre gênero e os grupos de depressão, sendo maiores em mulheres. 

table = sm.stats.Table(crosstab)
table.standardized_resids


def cramers_v(cross_tabs):

    from scipy.stats import chi2_contingency

    data = np.array([[1, .1, .3, .5],
       [2, .07, .21, .35],
       [3, .06, .17, .29],
       [4, .05,.15,.25],
       [5, .04, .13, .22]])
    sizes = pd.DataFrame(data, columns=['Graus de Liberdade', 'Efeito Pequeno', 'Efeito Médio', 'Efeito Grande']) 
    
    chi2 = chi2_contingency(cross_tabs)[0]
    n = cross_tabs.sum().sum()
    dof = min(cross_tabs.shape)-1
    v = np.sqrt(chi2/(n*dof))
    print(f'V de Cramer = {v}')
    print(f'Graus de liberdade do V de Cramer = {dof}')
    print(f'\nClassificação do V de Cramer\n{sizes}\n')

cramers_v(crosstab)

#V de Cramer = 0.07748197098312164
# Graus de liberdade do V de Cramer = 1

#As médias de idade são as mesmas pros três grupos de depressão?
grafico_boxplot_grp(df2, "RIDAGEYR", "Idade")

from scipy.stats import f_oneway

df_aux = df2[["phq_grp2", "RIDAGEYR"]].dropna()

stat, p = f_oneway(df_aux[(df_aux.phq_grp2 == "Sem sintomas")]["RIDAGEYR"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n leves")]["RIDAGEYR"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n moderados-severos")]["RIDAGEYR"])

print('stat=%.3f, p=%.3f' % (stat, p))

# O valor de p do teste foi p = 0,053. Nesse caso, não conseguimos detectar uma diferença estatisticamente significativa entre os grupos de depressão com relação à media de idade.

# As médias de renda familiar são as mesmas pros três grupos de depressão?

grafico_boxplot_grp(df2, "INDFMINC", "Renda Anual Familiar (US$)")

from scipy.stats import f_oneway

df_aux = df2[["phq_grp2", "INDFMINC"]].dropna()

stat, p = f_oneway(df_aux[(df_aux.phq_grp2 == "Sem sintomas")]["INDFMINC"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n leves")]["INDFMINC"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n moderados-severos")]["INDFMINC"])

print('stat=%.3f, p=%.3f' % (stat, p))

# stat=49.960, p=0.000

from statsmodels.stats.multicomp import pairwise_tukeyhsd

label = {"phq_grp2":
         {"Sem sintomas": "Sem sintomas",
          "Sintomas \n leves": "Sintomas leves",
          "Sintomas \n moderados-severos": "Sintomas moderados-severos"}}

df_aux = df_aux.replace(label)

tukey = pairwise_tukeyhsd(df_aux["INDFMINC"],
                  df_aux['phq_grp2'],
                  alpha = 0.05)

print(tukey)

# Teste 1: Grupo 0 (Sem sintomas) x Grupo 1 (sintomas leves) -> p = 0.001, A renda média do grupo Sem sintomas é diferente da renda média do grupo de sintomas leves.
# Teste 2: Grupo 0 (Sem sintomas) x Grupo 2 (sintomas moderados-severos) -> p = 0.001. A renda média do grupo Sem sintomas é diferente da renda média do grupo de sintomas moderados-severos
# Teste 3: Grupo 1 (sintomas leves) x Grupo 2 (sintomas moderados-severos) -> p = 0.001. A renda média do grupo sintomas leves é diferente da renda média do grupo sintomas moderados-severos

# As médias de horas semanais de exercícios são as mesmas para os três grupos de depressão?

grafico_boxplot_grp(df2, "PAG_HW", "Exercícios físicos (horas/ semana)")

from scipy.stats import f_oneway

df_aux = df2[["phq_grp2", "PAG_HW"]].dropna()

stat, p = f_oneway(df_aux[(df_aux.phq_grp2 == "Sem sintomas")]["PAG_HW"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n leves")]["PAG_HW"],
                   df_aux[(df_aux.phq_grp2 == "Sintomas \n moderados-severos")]["PAG_HW"])

print('stat=%.3f, p=%.3f' % (stat, p))
                   
# stat=12.983, p=0.000

from statsmodels.stats.multicomp import pairwise_tukeyhsd

df_aux = df_aux.replace(label)

tukey = pairwise_tukeyhsd(df_aux["PAG_HW"],
                  df_aux['phq_grp2'],
                  alpha = 0.05)

print(tukey)

# Teste 1: Grupo 0 (Sem sintomas) x Grupo 1 (sintomas leves) -> p = 0.273. O tempo médio de exercício físico por semana do grupo Sem sintomas não é diferente do tempo médio de exercício físico por semana do grupo de sintomas leves
# Teste 2: Grupo 0 (Sem sintomas) x Grupo 2 (sintomas moderados-severos) -> p = 0.001. O tempo médio de exercício físico por semana do grupo Sem sintomas é diferente do tempo médio de exercício físico por semana do grupo de sintomas moderados-severos
# Teste 3: Grupo 1 (sintomas leves) x Grupo 2 (sintomas moderados-severos) -> p = 0.001. O tempo médio de exercício físico por semana do grupo sintomas leves é diferente do tempo médio de exercício físico por semana do grupo sintomas moderados-severos