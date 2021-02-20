These two scripts are used to make the dataset in required format for training the SinSRL-Direct Annotator.

1. The pred_model.py script is used to generate training data for the predicate model of SinSRL: Direct Annotator (see the Pred_model directory ). 
2. The srl_model.py script is used to generate training data for the SRL model of SinSRL: Direct Annotator(see the SRL_model directory ). 

You are given a sample input file ( input.json ) and a sample output file (input.txt) of each script. Both scripts generate BIO tagged data. You can remove BIO tags if you don't need.