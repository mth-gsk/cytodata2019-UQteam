import json
import os

def save_model(model, name, location="saved_models"):
        location = os.path.join(location, name)
        if not os.path.exists(location):
                os.makedirs(location)
        json_string = model.to_json()
        model.save_weights(os.path.join(location, 'weights.h5'))
        with open(os.path.join(location, 'model.json'), 'w') as outfile:
                json.dump(json_string, outfile)