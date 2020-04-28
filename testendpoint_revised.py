"""
The following code uses the 100k movie lens example to make inference against an endpoint deployed on Sakemaker
to determine if a movie is suitable for a user in the dataset.
This example uses two of the users in test set as an example to determine a score for the movie user combination
"""

import boto3, csv, json
import numpy as np
from scipy.sparse import lil_matrix

nbUsers = 943
nbMovies = 1682

moviesByUser = {}
for userId in range(nbUsers):
    moviesByUser[str(userId)] = []

def loadDataset(filename, lines, columns):
    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((lines, columns)).astype('float32')
    # Labels are stored in a vector
    Y = []
    line = 0
    with open(filename, 'r') as f:
        samples = csv.reader(f, delimiter='\t')
        for userId, movieId, rating, timestamp in samples:
            X[line, int(userId) - 1] = 1
            X[line, int(nbUsers) + int(movieId) - 1] = 1
            if int(rating) >= 4:
                Y.append(1)
            else:
                Y.append(0)
            line = line + 1

    Y = np.array(Y).astype('float32')
    return X, Y


nbRatingsTrain = 90570
nbRatingsTest = 9430
nbFeatures = nbUsers + nbMovies

# using the ua.test dataset for the 100k movielens
X_test, Y_test = loadDataset('ua.test', nbRatingsTest, nbFeatures)
X_test[1000:1001].toarray()

data = X_test[1000:1002].toarray()
print (X_test[1000:1002])
print (X_test[1000:1002].toarray())

# this code serialises the data for use by the sagemaker API, in this case is expands the sparse matrix to a dense version
# see https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html for more examples of dense ans spare formats
def fm_serializer(data):
    js = {'instances': []}
    for row in data:
        js['instances'].append({'features': row.tolist()})
    # print js
    return json.dumps(js)


payload = fm_serializer(data)

runtime_client = boto3.client('sagemaker-runtime', region_name='us-east-2', aws_access_key_id='AKIAJZUFKCJLNG6TDUSQ',
                              aws_secret_access_key='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')

# the endpoint traing on the 100k dataset and deployed to AWS
endpoint_name = 'factorization-machines-2020-04-27-21-19-29-729'
response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                          ContentType='application/json',
                                          Accept='application/json',
                                          Body=payload)

# the scoring returned for the datapoints for the user and movie
print(response)
print(response['Body'].read())