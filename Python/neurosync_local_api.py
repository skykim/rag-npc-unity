from flask import request, jsonify
import numpy as np
import torch
import flask

from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.model import load_model
from utils.config import config

app = flask.Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    audio_bytes = request.data
    generated_facial_data = generate_facial_data_from_bytes(audio_bytes, blendshape_model, device, config)
    generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data

    return jsonify({'blendshapes': generated_facial_data_list})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5005)
