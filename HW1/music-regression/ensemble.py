import pandas as pd

df_CNN_melspec = pd.read_csv('submission/CNN_submission_melspec.csv')
df_CNN_melspecbest = pd.read_csv('submission/CNN_submission_melspec_best0.00900.csv')
df_CNN_mfcc = pd.read_csv('submission/CNN_submission_mfcc.csv')
df_MLP = pd.read_csv('submission/MLP_submission.csv')
df_LSTM = pd.read_csv('submission/submission_LSTM.csv')
df_ensemble = df_MLP
#score_avg  = (df_CNN_melspec['score'] + df_MLP['score']) / 2
#score_avg = (df_CNN_melspec['score'] + df_CNN_melspecbest['score'] + df_CNN_mfcc['score']) / 3
#score_avg  = (df_CNN_melspecbest['score'] + df_CNN_mfcc['score']) / 2
score_avg  = (df_CNN_melspec['score'] + df_CNN_melspecbest['score']) / 2
#score_avg  = (df_CNN_melspec['score'] + df_LSTM['score']) / 2

df_ensemble['score'] = score_avg

#df_ensemble.to_csv('submission/CNN_melspecmfcc_submission.csv', index=False)
#df_ensemble.to_csv('submission/CNN_melspec+LSTM_submission.csv', index=False)
#df_ensemble.to_csv('submission/CNN_melspecmfcc+MLP_submission.csv', index=False)
df_ensemble.to_csv('submission/CNN_melspecavg_submission.csv', index=False)