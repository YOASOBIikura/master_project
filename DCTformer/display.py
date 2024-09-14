import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results/exchange_ns_TDformer_random_modes64_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_1/result.csv')
truth = data[['truth']]
prediction = data[['prediction']]

plt.figure()
plt.plot(truth, label='GroundTruth', lw=2)
plt.plot(prediction, label='NPformer', lw=2)
plt.legend()
plt.show()