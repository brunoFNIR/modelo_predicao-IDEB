
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('br_inep_ideb_brasil.csv')

base_publica = base.loc[base['rede'] == 'publica']
base_privada = base.loc[base['rede'] == 'privada'] 


base_privada = base_privada.sort_values(by='ano')
base_publica = base_publica.sort_values(by='ano')

# Grafico de barras IDEB - Rede publica

categorias = base_publica.ano.loc[base_publica['ensino'] == 'fundamental']
valores = base_publica.ideb.loc[base_publica['ensino'] == 'fundamental']

fig, ax = plt.subplots()

ax.bar(categorias, valores)
plt.show()

# Grafico de barras IDEB - Rede pprivada

categorias = base_privada.ano
valores = base_privada.ideb

plt.bar(categorias, valores)
plt.xlabel('Ano')
plt.ylabel('IDEB')
plt.title('IDEB - Rede privada ao longos dos Anos')