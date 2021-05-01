# Dual_ALM_RNN

Pytorch v1.0 should be used to run these codes (other versions have not been tested).

To create a dataset, run the following:
```
python dual_alm_rnn_main.py --generate_dataset
```

To train a model, for example, a modular RNN, run the following:
```
python dual_alm_rnn_main.py --train_type_modular
```

To plot a time trace of CD projection, run the following:
```
python dual_alm_rnn_exp.py --plot_cd_traces
```

You can play with different parameters through dual_alm_rnn_configs.json.
