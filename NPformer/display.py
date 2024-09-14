import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('prediction/exchange_TDformer_custom_ftM_sl20_ll10_pl20_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/res.csv')
truth = data[['truth']]
prediction = data[['prediction']]

plt.figure(300, figsize=(30, 8))
plt.plot(truth, label='GroundTruth', lw=2)
plt.plot(prediction, label='NPformer', lw=2)
plt.legend()
plt.show()