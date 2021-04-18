import flask
from flask import request, jsonify
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
import yaml


app = flask.Flask(__name__)
app.config["DEBUG"] = True

# load intents to bert
prefix_q = '##### **Q:** '
topk = 5

yaml_file = open("./data/nlu.yml")
parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

questions = []
print("Loading intents ...")
for item in parsed_yaml_file.get('nlu'):
    question = item.get('examples').replace('-', '').split('\n ')
    questions = questions + question
print('%d Intent loaded.., avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

bc = BertClient(port=8001, port_out=8002)
doc_vecs = bc.encode(questions)


@app.route('/', methods=['GET'])
def home():
    return '''<p>service is running...</p>'''


# A route to return all of the available entries in our catalog.
@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)

@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'q' in request.args:
        query = request.args['q']
    else:
        return "Error: No q field provided. Please specify an q."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
    for idx in topk_idx:
        print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))
        results.append(questions[idx])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

app.run()