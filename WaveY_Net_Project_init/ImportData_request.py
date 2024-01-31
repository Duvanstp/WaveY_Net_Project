import io
import time
import requests
import numpy as np

start_time = time.time()
print('Init request')

response = requests.get('http://metanet.stanford.edu/static/search/waveynet/data/train_ds.npz')
response.raise_for_status()
train_data = np.load(io.BytesIO(response.content))

train_structures                = train_data['structures']
train_Hy_fields                 = train_data['Hy_fields']
train_dielectric_permittivities = train_data['dielectric_permittivities']

print('End request')
end_time = time.time()

print('Time request: ', end_time-start_time, ' seconds')