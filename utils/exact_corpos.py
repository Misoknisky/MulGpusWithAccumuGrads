#coding=utf-8
import json
trainFile="data/alldata/trainset/all.train.json"
devFile="data/alldata/trainset/all.dev.json"
testFile="data/alldata/trainset/all.test.json"
fileNames=[trainFile,devFile,testFile]
for filename in fileNames:
    with open(filename,"r") as f:
        for line in f:
            sample=json.loads(line.rstrip())
            for paragraphs in sample["documents"]:
                passages=paragraphs["segmented_passage"]
                for p in passages:
                    print(p)