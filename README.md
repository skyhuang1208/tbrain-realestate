# TBrain - E.SUN House Price Prediction #
4th place solution to the TBrain competition - E.SUN AI Open Competition Summer 2019 - House Price Prediction ([玉山人工智慧公開挑戰賽2019夏季賽 - 台灣不動產AI神預測](https://tbrain.trendmicro.com.tw/Competitions/Details/6))

### Contributors ###
- Chen-Hsi (Sky) Huang ([github.com/skyhuang1208](http://github.com/skyhuang1208))
- Louis Yang ([github.com/louis925](http://github.com/louis925))

### Achievement ###
We won 4th place out of 766 teams (top 0.5%) with private leaderboard score 6210.877.

<img src="doc/Leaderboard.png" alt="Leaderboard" width="600"/>

### Documentation ###
- [Modeling description (in Chinese)](https://github.com/skyhuang1208/tbrain-realestate/blob/master/doc/建模說明文件_Sky.pdf)
- [Presentation (in Chinese)](https://docs.google.com/presentation/d/1P-N77IkxL-ps-dCR02iWHv2dclDLPBRwHjGHy4CwK-U/edit?usp=sharing)
- [Local leaderboard](https://docs.google.com/spreadsheets/d/16IqWaDjwOKHsVHiRDIBNgdJejel-pppoaUZHPcP0gWw/edit?usp=sharing)

### Structure ###
- `dataset` - place input dataset `train.csv` and `test.csv`
  - `gen_5_fold_cv.ipynb` - generate 5-fold CV dataset
- `eda-and-exp/` - contains notebooks for exploratory data analysis and various experiments
  - `eda*` - exploratory data analysis
  - `exp*` - experiments
- `model-<model_number>-build-<model_name>.ipynb/.py` - parameters search with small training step for single model
- `model-<model_number>-predict-<model_name>.ipynb/.py` - complete single model training process and prediction. Output single model CV and test set prediction for stacking
- `stack_<stack_method>_<stacking_model_number>_<model_numbers_used>.ipynb` - stacking model from single models in `<model_numbers_used>`. Output to final test set prediction for submission.
- `feature_engineering.py` - label encoders and feature scalers
- `keras_get_best.py` - early stop callback for keras
- `utilities.py` - utility functions including scoring, feature processing, ..., etc
- `vars_03.py` - feature importance computed by experiments for feature selection
- `loss_exp.ipynb` - experiment on the smooth hit rate loss function
- `price_quantizer*.ipynb` - quantize predicted price used by stack 16 and 18

### How it work ###
