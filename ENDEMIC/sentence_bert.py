"""Sentence Bert"""

# !pip install sentence_transformers
import pickle
import json
# import args
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# roberta-large-nli-stsb-mean-tokens

'''Variable to dedice file'''
type = 2

'''Location for output folder'''
output_file  = "data/graph_new/debug/ek"

#Choose 1 for fake and gen files 2 for unlabelled files
if type == 1:
    file_path =r'data/graph_new/debug/data_gen_fake_2629.txt'
    #a similar file for fake was given to you keep it here.
    file_text = r'data/graph_new/debug/Gen_fake_uipath_2629.txt'    #this file has query and title urls mapped. need it for mapping scores with query
if type ==2:
    file_path = r'data/graph_new/debug/unlabelled_next.txt'  # , 'r') as file:

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

print(tokens_without_sw)
print(" ".join(tokens_without_sw))

# "emojis"
import re
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
print(emoji_pattern.sub(r'', text)) # no emoji

if type == 1:
    with open(file_text, 'r') as file:
        for i in file:
            x = json.loads(i)
    with open(file_path, 'r',encoding="utf8") as file:
        for i in file:
            print(i[:10])
            list_str = i.split("article_")
    
    print("LOADED")

    list_str.pop(0)
    corpus_ele=[]
    print("Number of scrapped articles: ", len(list_str))
    for n in range(0,len(list_str)) :
        print(n, ":")
        try:
            query = x[n]['query']
            corpus_1=[]
            sentences=[]
            corpus_1 += [list_str[n].split(".")]
            for i in range(0,len(corpus_1)):
                    corpus_X = []
                    corpus_1[i] = list(filter(lambda x: x !="", corpus_1[i]))
                    corpus_1[i] = list(filter(lambda x: x !=" ", corpus_1[i]))
                    corpus_1[i] = list(filter(lambda x: len(x.split(" "))>4, corpus_1[i]))
                    
                    for ele in corpus_1[i]:
                        
                        ele = ele.replace("  ","")
                        ele = ele.replace("\n","")
                        ele = ele.replace("\n \n ","")
                        ele = ele.replace("': [['","")
                        ele = ele.replace("']]}{'", "")
                        ele = ele.replace("': [['", "")
                        ele = ele.replace("']]},{'",'')
                        ele = ele.replace('"]]','')
                        ele = ele.replace("[['",'')

                        ele = ele.strip()
                        ele = emoji_pattern.sub(r'', ele)
                        text_tokens = word_tokenize(ele)
                        
                        ele = " ".join(tokens_without_sw)
                        corpus_X.append(ele)
                        
                    corpus_ele.append(corpus_X)
            print(corpus_ele[0],"n corpus")
            if corpus_ele[n]!=[]:
                corpus_embeddings = model.encode(corpus_ele[n], convert_to_tensor=True)
                from sentence_transformers import SentenceTransformer, util
                embedder = model
                for ind,query in enumerate([query]):
                    query_embedding = embedder.encode(query, convert_to_tensor=True)
                    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
                #     print(cos_scores,"----")
                    cos_scores = cos_scores.cpu()
                    top_results = list(filter(lambda x: cos_scores[x] > 0.7, range(0,len(cos_scores))))
                    print(top_results)
                    print("\n\n======================\n\n", i)
                    print("Query:", query)
                    print("\nTop 5 most similar sentences in corpus:")
                    for idx in top_results:
                        print(corpus_ele[n][idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
                        sentences.append(corpus_ele[n][idx].strip())
                if len(sentences)<1:
                    print("less sentences",n)
                else:
                    sentence_embeddings = model.encode(sentences)
                    # print("we are here")  `
                    # break
                    with open(output_file +'/filename'+str(n)+'.pkl','wb') as f: pickle.dump(sentence_embeddings, f)
        except Exception as e:
            print(e)
            pass

if type == 2:

    with open(file_path, 'r') as file:
        data = json.load(file)
    sentence_vocab = []
    x = data
    for n in range(0, len(x)):
        if x[n] != [] or x[n] != None:
            try:
                query = x[n]['query']
            except Exception as e:
                query = ""
            queries = [query]
            corpus_1 = []
            sentences = []
            try:
                print(len(x[n]['top5']), "len top 5")
                for n_n in range(0, len(x[n]['top5']) - 1):
                    print(n_n, "n_n")
                    #         print(x[n]['top5'])
                    #             print(x[n]['top5'][n_n])
                    corpus = x[n]['top5'][n_n]['article_' + str(n_n + 1)]
                    corpus_1 += [corpus.split(".")]
                    corpus_ele = []
                    for i in range(0, len(corpus_1)):
                        corpus_X = []
                        corpus_1[i] = list(filter(lambda x: x != "", corpus_1[i]))
                        corpus_1[i] = list(filter(lambda x: x != " ", corpus_1[i]))
                        corpus_1[i] = list(filter(lambda x: len(x.split(" ")) > 4, corpus_1[i]))
                        for ele in corpus_1[i]:
                            ele = ele.replace("  ", "")
                            ele = ele.replace("\n", "")
                            ele = ele.replace("\n \n ", "")
                            ele = ele.strip()
                            ele = emoji_pattern.sub(r'', ele)
                            text_tokens = word_tokenize(ele)
                            ele = " ".join(tokens_without_sw)
                            corpus_X.append(ele)
                        corpus_ele.append(corpus_X)
                    corpus_embeddings = model.encode(corpus_ele[n_n], convert_to_tensor=True)
                    from sentence_transformers import SentenceTransformer, util

                    embedder = model
                    for ind, query in enumerate(queries):
                        query_embedding = embedder.encode(query, convert_to_tensor=True)
                        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
                        #     print(cos_scores,"----")
                        cos_scores = cos_scores.cpu()
                        top_results = list(filter(lambda x: cos_scores[x] > 0.7, range(0, len(cos_scores))))
                        print("\n\n======================", i)
                        print("Query:", query)
                        print("\nTop 5 most similar sentences in corpus:")

                        for idx in top_results:
                            print(corpus_ele[n_n][idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
                            sentences.append(corpus_ele[n_n][idx].strip())

                            if len(sentences)<1:
                                print("less sentences",n)
                            else:
                                sentence_embeddings = model.encode(sentences)
                                with open(output_file+'/unlabelled'+str(n)+'.pkl','wb') as f:
                                    pickle.dump(sentence_embeddings, f)

            except Exception as e:
                print(e)
                pass
