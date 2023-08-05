def lang_detect(text):
  import numpy as np
  import re
  import string
  import pickle
  import os 

  translate_table = str.maketrans('', '', string.punctuation)
  global lrLangDetectmode1
  file_path = os.path.join(os.path.dirname(__file__), 'LRModel.pck1')
  lrLangDetectFile = open(file_path, 'rb')
  #lrLangDetectFile=open('LRModel.pck1','rb')
  lrLangDetectmode1=pickle.load(lrLangDetectFile)
  lrLangDetectFile.close()
  
  
  text=" ".join(text.split())
  text=text.lower()
  text=re.sub(r"\d+", "", text)
  text =text.translate(translate_table)
  pred=lrLangDetectmode1.predict([text])
  prob=lrLangDetectmode1.predict_proba([text])
  return pred[0]
