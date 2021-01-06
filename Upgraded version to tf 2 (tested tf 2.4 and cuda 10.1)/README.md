# direct-sin-srl

1. First satisfy the following requirements
    
    `python 3.6`  
    `tensorflow 2.2`    
    `tensorflow-addon`  
    `gensim 3.4.0`  
    ```Tested with cuda 10.1```
2. Training data should be in following directories

    The predicate identifier - `/data/predIdData`  
    The srl tagger - `/data/srlIdData`

3. Fasttext word embedding model should be in `/data` directory

4. To train the model execute following script

    The predicate identifier - `python train.py predId train`  
    The srl tagger - `python train.py srlId train`

5. If training was interrupted. It can be continued again using following command
    
    `python train.py predId retrain {{modelname}}`  
    `python train.py srlId retrain {{modelname}}`  
    model name here should be the directory name that saves checkpoints for the model that should continue with training. They can be found at 
    `root/results/test/predIdData/model.weights`  
    `root/results/test/srlIdData/model.weights`  
    
6. Training parameters can be configured using `parameters.json` file which available in the project root directory
    
7. To get the predictions from each trained models -

    predicate identifier - `python individualModelpredict.py predId {{modelname}}`  
    srl tagger - `python individualModelpredict.py srlId {{modelname}}`  
               
8.To get the final prediction for a given Sinhala sentence 
    
    `python finalPredict.py {{predId model name}} {{srlId model name}}`  

9. logs are stored in `/results` directory


### When run in COLAB
execute following code snippet inside the browser console to remain connected the colab session

```
function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-connect-button").click() 
}
setInterval(ClickConnect,60000)

```
