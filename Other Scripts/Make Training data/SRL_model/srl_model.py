import json
import sys
import os

special_character = "*"
cwd = os.getcwd() ## get current workin directory

with os.scandir(cwd) as it:
    count = 0
    for entry in it:
        if entry.name.endswith(".json") and entry.is_file():
            print(entry.name)
            filename= entry.name.split(".")[0]
            with open(filename+'.txt', "w", encoding="utf-8") as output: # we can use 'a' insted of 'w' of want to append 
                output.write("-DOCSTART-"+" O\n\n")
                with open(entry.name, encoding='utf-8') as fp: 
                    data = json.load(fp)
                    for sentence in data:
                        if len(sentence) > 0:
                            count+=1
                        try:
                            noOfRound=sentence[0]['frame'].count(',')+1
                            for round in range(noOfRound): 
                                duplicate_keeper=set([])     
                                for word in sentence:                          
                                    frame= word['frame'].strip()[1:-1].split(", ")
                                    text = word['text'].strip()
                                    item =  frame[round]      
                                    if("." in item):
                                        for part in text.split(" "):                                
                                            if(item in duplicate_keeper):
                                                output.write(part+"-I-"+item+"  I-"+item+"\n")                     
                                            else:
                                                output.write(part+"-B-"+item+"  B-"+item+"\n")
                                                duplicate_keeper.add(item)
                                                
                                    elif (text in special_character):  
                                        continue   
                                    elif("ARG" in item):
                                        BIOTagRemovedItem=item[2:]
                                        for part in text.split(" "): 
                                            if(BIOTagRemovedItem in duplicate_keeper):
                                                output.write(part+" I-"+BIOTagRemovedItem+"\n")                     
                                            else:
                                                output.write(part+" B-"+BIOTagRemovedItem+"\n")
                                                duplicate_keeper.add(BIOTagRemovedItem)
                                    else:

                                        for part in text.split(" "):
                                            output.write(part+" O\n")

                                if data.index(sentence) != len(data)-1:
                                    output.write("\n")
                        except :
                            e = sys.exc_info()[0]
                            print(e)
                            print(sentence)
                            continue

                fp.close 
            output.close
    print ("Total successful sentences: "+str(count))
