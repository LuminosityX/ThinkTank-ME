
import sys

sys.path.append("./BERTopic")
from bertopic import BERTopic

import hdbscan
from umap import UMAP
import argparse
import json
import os
from tqdm import tqdm 
import pandas as pd
import numpy as np

RANDOM_STATE = 2024

parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbors', type=int, default=50,
                    help='how UMAP balances local versus global structure in the data')
parser.add_argument('--n_components', type=int, default=32,
                    help='n_dimension that UMAP will reduce to')
parser.add_argument('--min_dist', type=float, default=0.0,
                    help='point min distance after UMAP')
parser.add_argument('--min_topic_size', type=int, default=20,
                    help='minimum number of atomic events in a complex event after HDBSCAN')
parser.add_argument('--top_n_words', type=int, default=20,
                    help='topic n words for each cluster (complex event)')
parser.add_argument('--input_path', type=str, default='./data_mideast/',
                    help='output path of bertopic model')
parser.add_argument('--output_path', type=str, default='./data_mideast/cluster/',
                    help='output path of bertopic model')
parser.add_argument('--time_weight', type=float, default=1.0,
                    help='time feature weight')

args = parser.parse_args("")
# args = parser.parse_args()
args.umap_emb_path = args.input_path
input_path = args.input_path

curr_setting = 'time_t{}_n{}_m{}'.format(args.time_weight, args.n_neighbors, args.min_topic_size)
output_path = args.output_path + curr_setting + '/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

print('start loading documents')
docs = json.load(open(input_path + "news_article_list.json", 'r', encoding='latin-1'))
print('end loading documents, length: ')
print(len(docs))

print('start loading document embeddings')
doc_embs = np.load(input_path + 'doc_embs.npy')
print('end loading document embeddings, shape: ')
print(doc_embs.shape)

print('start loading time embeddings')
time_embs = np.load(input_path + 'time_embs.npy')
time_embs = time_embs * args.time_weight
print('end loading time embeddings with weight {}, shape: '.format(args.time_weight))
print(time_embs.shape)

umap_embs = np.load(args.umap_emb_path + f'umap_embeddings_nn{args.n_neighbors}_nc{args.n_components}.npy')
print('Umap embedding loaded from: ' + args.umap_emb_path + f'umap_embeddings_nn{args.n_neighbors}_nc{args.n_components}.npy, shape: ')
print(umap_embs.shape)

umap_model = UMAP(
    n_neighbors=args.n_neighbors, # how UMAP balances local versus global structure in the data.
    n_components=args.n_components, # n_dimension reduced to
    min_dist=args.min_dist,
    metric='cosine',
    low_memory=False,
    random_state=RANDOM_STATE)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=args.min_topic_size, # minimum number of atomic events in a complex event
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True)

bertopic_model = BERTopic(
    language = "english",
    top_n_words = args.top_n_words,
    calculate_probabilities = False, # onlyu calculate the prob that a document belongs to the assigned CE
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    verbose = True,
)

topic_model = bertopic_model.fit_time_umap(docs, embeddings=doc_embs, time_embs=time_embs, umap_embs=umap_embs, save_path=output_path)
# topic_model = bertopic_model.fit_transform(docs, embeddings=doc_embs)

#print(topic_model.get_topic_info())
topic_info = topic_model.get_topic_info()
topic_info.to_csv(path_or_buf=f'{output_path}/topic_info.csv', sep='\t', index=False)

n_topics = len(topic_model.get_topics()) - 1
print(f"n_topics: {n_topics}")

event_df = pd.read_csv('../MidEast.csv', sep=',', dtype='string')
md5_list = json.load(open(input_path + 'md5_list.json'))
md52topicid = {}
for idx, topicid in enumerate(topic_model.topics_):
    md52topicid[md5_list[idx]] = topicid
event_df['Topic'] = [md52topicid[_] for _ in event_df['Md5']]
md52nday = json.load(open(input_path + 'md52nday.json'))
event_df['Nday'] = [md52nday[_] for _ in event_df['Md5']]

event_df.to_csv(path_or_buf=f'{output_path}/t1_m20_raw.csv', sep='\t', index=False)

results = event_df.groupby(['Topic']).agg({'Nday':['count', 'max','min','mean']})
results.columns = ['n_events', 'nday_max', 'nday_min', 'nday_mean']
results['nday_range'] = results['nday_max'] -results['nday_min']

results.to_csv(path_or_buf=f'{output_path}/result.csv', sep='\t', index=False)

print(results.loc[0:, :].mean())
print(results.loc[0:, :].max())
print(results.loc[0:, :].min())