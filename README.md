# direct-sin-srl

python 3.6

tensorflow 1.4.0

data to train the predicate identifier - in /data/predIdData

data to train the srl tagger - in /data/srlIdData


put a fasttext word embedding model into /data directory


to train the predicate identifier - python train.py predId (Until making hparams tuning, you can change configurations in predIdConfig.py file)

to train the srl tagger - python train.py srlId  (Until making hparams tuning, you can change configurations in srlPredConfig.py file)


to see the predictions from each trained models -


predicate identifier - python individualModelpredict.py predId
          
srl tagger - python individualModelpredict.py srlId
          
          
to get the final prediction for a given sinhala sentence - python finalPredict.py
