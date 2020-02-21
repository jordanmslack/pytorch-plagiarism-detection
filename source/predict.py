import os
import numpy as np
import torch
from six import BytesIO
from model import BinaryClassifier


NP_CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    
    """Load the PyTorch model from the `model_dir` directory."""
    
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    
    return model


def input_fn(serialized_input_data, content_type):
    
    print('Deserializing the input data.')
    
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    
    print('Serializing the generated output.')
    
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        
        return stream.getvalue(), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    model.eval()

    out = model(data)
    out_np = out.cpu().detach().numpy()
    out_label = out_np.round()

    return out_label