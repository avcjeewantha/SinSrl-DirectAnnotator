import json
import sys
import os

special_character = "*"
cwd = os.getcwd() ## get current workin directory

with os.scandir(cwd) as it:
    for entry in it:
        if entry.name.endswith(".json") and entry.is_file():
            print(entry.name)
            filename= entry.name.split(".")[0]
            with open(filename+'.txt', "w", encoding="utf-8") as output: # we can use 'a' insted of 'w' of want to append 
                output.write("-DOCSTART-"+" O\n\n")
                with open(entry.name, encoding='utf-8') as fp: 
                    data = json.load(fp)
                    for sentence in data:  
                        if not sentence:
                            continue          
                        try:     
                            for word in sentence:                  
                #                print(word['frame'].strip())                    
                                frame= word['frame'].strip()
                                text = word['text'].strip()                       
                                if("." in frame):
                                    count=0
                                    sense = [sense for sense in frame[1:-1].split(", ") if "." in sense][0] 
                                    for part in text.split(" "):                                
                                        if(count==0):
                                            output.write(part+" B-"+sense+"\n")
                                            count=count+1                       
                                        else:
                                            output.write(part+" I-"+sense+"\n") 
                                elif (text in special_character):  
                                    continue    
                                else:
                                    for part in text.split(" "):
                                        output.write(part+" O\n")
                            
                            if data.index(sentence) != len(data)-1:
                                output.write("\n")
                        except:                
                            e = sys.exc_info()[0]
                            print(e)
                            print(sentence)   
                            continue    

                fp.close 
            output.close
#             print(word['frame'].strip()[1:-1].split(", ")) 
