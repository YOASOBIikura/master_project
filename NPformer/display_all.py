import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('prediction/exchange_TDformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/res.csv')
clo = data[['data']]

plt.figure(dpi=400, figsize=(30, 8))
plt.plot(clo)
plt.show()