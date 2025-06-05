import argparse
import json
import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from datetime import datetime

def main(args, events):
    event_md5_list = list(np.unique(events['Md5']))
    #json.dump(event_md5_list, open(args.output_path + 'md5_list_1.json', 'w'), indent=4)
    print('event md5 list length: ' + str(len(event_md5_list)))

    # generate the list form of news article content for clustering
    with open(args.output_path + "news_article_list.json", mode="w", encoding='latin-1') as f:
        json.dump(events["News_article"], f, indent=4)

    # generate the 'date2nday.json' file which dates corresponding to the nday ID
    dates = list(np.unique(events['Date']))
    print('event dates: ' + str(len(dates)))
    mindate = datetime.strptime(dates[0], '%Y-%m-%d')
    date2nday = {}
    for date in dates:
        nday = (datetime.strptime(date, '%Y-%m-%d') - mindate).days
        date2nday[date] = nday
    json.dump(date2nday, open(args.output_path + 'date2nday.json', 'w'), indent=4)

    # generate the 'md52nday.json' file which MD5 corresponding to the nday ID
    md52nday = {}
    events['ndays'] = [date2nday[x] for x in events['Date']]
    for md5_i, ndays_i in tqdm(zip(events['Md5'], events['ndays']), total=len(events)):
        md52nday[md5_i] = ndays_i
    json.dump(md52nday, open(args.output_path + 'md52nday.json', 'w'), indent=4)

    # generate the 'md5_list.json' file
    # csv_ = []
    list_ = []
    for index_i, item_i in md52nday.items():
        # csv_.append({"Md5":index_i}) 
        list_.append(index_i)

    # df = pd.DataFrame(csv_, columns=["Md5"])
    # df.to_csv("./news_Md5.csv", index=False)

    with open(args.output_path + "md5_list.json", mode="w", encoding='latin-1') as f:
        json.dump(list_, f, indent=4)

    # generate the time embedding numpy file 
    time_features = []
    maxnday = max(events['ndays'])
    print(f"maxnday: {maxnday}")
    for md5 in event_md5_list:
        time_features.append(1.0 * md52nday[md5] / maxnday)
    time_embs = np.array(time_features)
    np.save(args.output_path + 'time_embs.npy', time_embs)
    print('time embeddings saved, shape: {}'.format(time_embs.shape))

    # pre-generate the text embedding numpy file 
    from sentence_transformers import SentenceTransformer

    # Pre-calculate embeddings
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedding_model.encode(events["News_article"], show_progress_bar=True)

    print("text embedding info:")
    print(type(embeddings)) 
    print(embeddings.shape)  

    np.save(args.output_path + "doc_embs.npy", embeddings)

    # obtaining a dimensionality reduction vector
    from umap import UMAP
    RANDOM_STATE =2024

    umap_model = UMAP(
        n_neighbors=args.n_neighbors, # how UMAP balances local versus global structure in the data.
        n_components=args.n_components, # n_dimension reduced to
        min_dist=args.min_dist,
        metric='cosine',
        n_jobs = 1,
        low_memory=False,
        random_state=RANDOM_STATE)

    print('start loading document embeddings')
    doc_embs = np.load(args.output_path + "doc_embs.npy")
    print('end loading document embeddings, shape: ')
    print(doc_embs.shape)

    umap_emb = umap_model.fit_transform(doc_embs)

    np.save(args.output_path + "umap_embeddings_nn50_nc32.npy", umap_emb)
    print('doc embeddings saved, shape: {}'.format(umap_emb.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../2_1_article_content/data_mideast/news_article_final.json',
                        help='input csv data')
    parser.add_argument('--output_path', type=str, default='./data_mideast/',
                        help='output path')
    
    # UMAP 
    parser.add_argument('--n_neighbors', type=int, default=50,
                    help='how UMAP balances local versus global structure in the data')
    parser.add_argument('--n_components', type=int, default=32,
                        help='n_dimension that UMAP will reduce to')
    parser.add_argument('--min_dist', type=float, default=0.0,
                        help='point min distance after UMAP')
    
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        print('Make new dir: ' + args.output_path)
    else:
        print('Dir exists: '+ args.output_path)

    print('start loading news_article data')
    news_article = json.load(open(args.input_path, 'r', encoding='latin-1'))
    print('end loading news_article data, number: ' +  str(len(news_article)))

    events_list = {"Md5": [], "Date": [], "News_article": []}
    for index_i, item_i in news_article.items():
        events_list["Md5"].append(index_i)
        events_list["Date"].append(item_i["news_date"])
        events_list["News_article"].append(item_i["news_article"])

    events_list["Md5"] = np.array(events_list["Md5"])
    events_list["Date"] = np.array(events_list["Date"])

    main(args, events_list)
