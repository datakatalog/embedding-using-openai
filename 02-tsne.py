import os
from dotenv import load_dotenv
from openai import OpenAI

# Load all .env variables
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise Exception("API key not found. Make sure OPENAI_API_KEY is set in your .env file.")
client = OpenAI(api_key=api_key)


articles = [
{"headline": "Economic Growth Continues Amid Global Uncertainty", "topic": "Business"},
{"headline": "Interest rates fall to historic lows", "topic": "Business"},
{"headline": "Scientists Make Breakthrough Discovery in Renewable Energy", "topic": "Science"},
{"headline": "India Successfully Lands Near Moon's South Pole", "topic": "Science"},
{"headline": "New Particle Discovered at CERN", "topic": "Science"},
{"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility", "topic": "Tech"},
{"headline": "Tech Giant Buys 49% Stake In AI Startup", "topic": "Tech"},
{"headline": "New Social Media Platform Has Everyone Talking!", "topic": "Tech"},
{"headline": "The Blues get promoted on the final day of the season!", "topic": "Sport"},
{"headline": "1.5 Billion Tune-in to the World Cup Final", "topic": "Sport"}
]

headline_text = [article['headline'] for article in articles]
headline_text

response = client.embeddings.create(
model="text-embedding-3-small",
input=headline_text
)
response_dict = response.model_dump()

for i, article in enumerate(articles):
    article['embedding'] = response_dict['data'][i]['embedding']
#print(articles[:2])
print(len(articles[0]['embedding']))
print(len(articles[5]['embedding']))

from sklearn.manifold import TSNE
import numpy as np
embeddings = [article['embedding'] for article in articles]
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))

import matplotlib.pyplot as plt
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
topics = [article['topic'] for article in articles]
for i, topic in enumerate(topics):
    plt.annotate(topic, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
#plt.show()

def create_embeddings(texts):
    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]

#print(create_embeddings(["Python is the best!", "R is the best!"]))
#print(create_embeddings("DataCamp is awesome!")[0])

from scipy.spatial import distance
import numpy as np
search_text = "computer"
search_embedding = create_embeddings(search_text)[0]
distances = []
for article in articles:
    dist = distance.cosine(search_embedding, article["embedding"])
    distances.append(dist)

min_dist_ind = np.argmin(distances)
print(articles[min_dist_ind]['headline'])


for headline, dist in zip([a['headline'] for a in articles], distances):
    print(f"{dist:.4f}  ->  {headline}")