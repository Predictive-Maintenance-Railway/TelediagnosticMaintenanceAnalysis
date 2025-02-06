# packages/frameworks:
#import numpy as np
#import pandas as pd


'''
PRIMA VERSIONE MIA SBAGLIATA:

 #   -we don't control for the misalignment, so timestamp is not used in the algorithm --> therefore we allow some duplications
#   -due to missing values in lat and lon, variable used in the algorithm are "complete_code" and "speed"

# prova algoritmo restringendo il dataset a 100 oss:
df_test = df9.iloc[34:51]

MD_obs = df_test[df_test['Monitor'] == 'MD'].index
print(MD_obs)

previous_md_obs = None
Real_Monitor = []
for k in range(len(MD_obs)):
	#if previous_md_obs is None:
	#    previous_md_obs = k
	for i in df_test.index.tolist():
		while i <= MD_obs[k]:
			if df_test.loc[i, 'Monitor'] == 'MD':
				Real_Monitor.append("MD")
			else:
				if df_test.loc[i, 'Monitor'] == 'MS' and df_test.loc[i, 'complete_code'] == df_test.loc[
					k, 'complete_code'] and df_test.loc[i, 'Speed'] == df_test.loc[k, 'Speed']:
					Real_Monitor.append("MS")
				else:
					Real_Monitor.append("MD")
else:
	for i in df_test.index.tolist():
		while previous_md_obs > i and i <= k:
			if df_test.loc[i, 'Monitor'] == 'MD':
				Real_Monitor.append("MD")
			else:
				if df_test.loc[i, 'Monitor'] == 'MS' and df_test.loc[i, 'complete_code'] == df_test.loc[
					k, 'complete_code'] and df_test.loc[i, 'Speed'] == df_test.loc[k, 'Speed']:
					Real_Monitor.append("MS")
				else:
					Real_Monitor.append("MD")
	previous_md_obs = k

print(Real_Monitor)

'''

# prova algoritmo restringendo il dataset a 50 oss:

df_test = df9.iloc[:50]

# GIORGIO:

MD_obs_test = df_test[df_test['Monitor'] == 'MD'].index
print(MD_obs_test)
print("\n")

appoggio = 0
Real_Monitor_giorgio = []
for k in range(len(MD_obs_test)):
	i = appoggio
	print(f"posizione MD_obs di riferimento:{k}")
	print(f"MD_obs di riferimento:{MD_obs_test[k]}")
	print(f"posizione osservazione che stiamo confrontanto rispetto a k:{i}")
	print("\n")
	while i <= MD_obs_test[k]:
		if df_test.loc[i, 'Monitor'] == 'MD':
			Real_Monitor_giorgio.append("MD")
		else:
			if df_test.loc[i, 'complete_code'] == df_test.loc[MD_obs_test[k], 'complete_code'] and df_test.loc[i, 'Speed'] == df_test.loc[MD_obs_test[k], 'Speed']:
				Real_Monitor_giorgio.append("MS")
			else:
				Real_Monitor_giorgio.append("MD")
		i += 1
	appoggio = i

#print(Real_Monitor_giorgio)

df_test["Real_Monitor_giorgio"] = Real_Monitor_giorgio


# GIULIA:
# prova allargando a k+1 (perchè non è sistematico che prima ci siano gli MS e poi gli MD)

MD_obs_test = df_test[df_test['Monitor'] == 'MD'].index
print(MD_obs_test)
print("\n")

appoggio = 0
Real_Monitor_giulia = []

for k in range(len(MD_obs_test)):
	i = appoggio
	print(f"posizione MD_obs di riferimento:{k}")
	print(f"MD_obs di riferimento:{MD_obs_test[k]}")
	print(f"prima posizione osservazione che stiamo confrontanto rispetto a k:{i}")
	print(f"ultima posizione osservazione che stiamo confrontanto rispetto a k:{MD_obs_test[k] + 1}")
	print("\n")
	if k == len(MD_obs_test) - 1:  # dato che questo dataset ridotto ha come ultima oss MD, c'è bisogno che all'ultima iterazione di MD_obs arrivi a considerare i <= MD_obs[k]
		while i <= MD_obs_test[k]:
			if df_test.loc[i, 'Monitor'] == 'MD':
				Real_Monitor_giulia.append("MD")
			else:
				if df_test.loc[i, 'complete_code'] == df_test.loc[MD_obs_test[k], 'complete_code'] and df_test.loc[i, 'Speed'] == df_test.loc[MD_obs_test[k], 'Speed']:
					Real_Monitor_giulia.append("MS")
				else:
					Real_Monitor_giulia.append("MD")
			i += 1
	else:
		while i <= MD_obs_test[k] + 1:  # per il resto invece confronta tutte le oss i-esime fino a quella successiva (compresa) rispetto a MD, a MD --> così che riusciamo a controllare sia l'ordine MD-MS che MS-MD
			if df_test.loc[i, 'Monitor'] == 'MD':
				Real_Monitor_giulia.append("MD")
			else:
				if df_test.loc[i, 'complete_code'] == df_test.loc[MD_obs_test[k], 'complete_code'] and df_test.loc[i, 'Speed'] == df_test.loc[MD_obs_test[k], 'Speed']:
					Real_Monitor_giulia.append("MS")
				else:
					Real_Monitor_giulia.append("MD")
			i += 1
	appoggio = i

#print(Real_Monitor_giulia)

df_test["Real_Monitor_giulia"] = Real_Monitor_giulia
df_test = df_test[["Time", "Monitor", "Real_Monitor_giorgio", "Real_Monitor_giulia", "complete_code", "Speed"]]