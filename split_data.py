import math
import numpy as np
import pickle as pkl
from tqdm import tqdm	
from IPython import embed



with open('data/simulation_data.pkl', 'rb') as f:
    raw_data = pkl.load(f)
    embed()

for i in tqdm(range(100)):
	pkl.dump(raw_data[i*100:(i+1)*100],open('data/raw_' + str(i) + '.pkl','wb'))