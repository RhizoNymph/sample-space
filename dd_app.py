# dash_app.py
import os
import pickle
import dash_bootstrap_components as dbc
from flask import Flask, send_from_directory

import dash
from scipy.io.wavfile import write
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from diffusers import DiffusionPipeline
import torch
from scipy.io.wavfile import write
import librosa
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from sklearn.manifold import TSNE

# Init code
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)

placeholder_plot = go.Figure()
placeholder_plot.layout.plot_bgcolor = 'black'
placeholder_plot.layout.paper_bgcolor = 'black'
placeholder_plot.layout.font = {'color': 'white'}

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY], prevent_initial_callbacks="initial_duplicate")

@app.callback(
    Output('audio-player', 'src'),
    [Input('tsne-output', 'clickData')]
)
def play_sound(clickData):
    if clickData is None:
        return ''

    # Base directory where audio files are stored
    base_dir = '/home/nymph/OpenSource/sample-space/audio/'

    # Extract the relative path from the absolute path in clickData
    full_path = clickData['points'][0]['text']
    if full_path.startswith(base_dir):
        relative_path = full_path[len(base_dir):]  # Remove the base directory part
        audio_url = f'/audio/{relative_path}'
        print("Serving audio from:", audio_url)  # Debug: print the URL being served
        return audio_url
    else:
        print("Error: Audio path does not start with base directory")
        return ''

@server.route('/audio/<path:path>')
def serve_audio(path):
    # Ensure this directory is correct and accessible by the Flask app
    audio_directory = '/home/nymph/OpenSource/sample-space/audio/'
    return send_from_directory(audio_directory, path)
    
# TSNE Code
def get_features(y, sr):	
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate(
        (np.mean(mfcc, 1), np.mean(delta_mfcc, 1), np.mean(delta2_mfcc, 1)))
    feature_vector = (feature_vector-np.mean(feature_vector)
                      ) / np.std(feature_vector)
    return feature_vector, y.mean()

def tsne_plot(model_id, perplexity=30, learning_rate=200):
    files = os.listdir('output')
    matching_files = [f for f in files if f.startswith(f"{model_id.replace('/', '-')}")]

    feature_vectors = []
    for f in tqdm(matching_files):
        y, sr = librosa.load("output/"+f)
        feat, avg_amp = get_features(y, sr)
        feature_vectors.append({"file": f, "features": feat, "avg_amp": avg_amp})

    tsne = TSNE(n_components=3, learning_rate=200, perplexity=50,
                verbose=1, angle=0.1, random_state=0)
    tsne = tsne.fit_transform(np.array([f["features"] for f in feature_vectors]))
    data = []

    for i, f in enumerate(feature_vectors):
        abspath = os.path.abspath(f['file'])
        file_name = os.path.basename(f['file'])
        data.append([abspath, file_name, tsne[i,0], tsne[i,1], tsne[i,2], f['avg_amp']])

    df = pd.DataFrame(data, columns =['path','file_name','x','y', 'z', 'avg_amp'])
    df.to_csv('feature_vectors.csv', index=False)

    f = go.FigureWidget([go.Scatter3d(x=df.x, y=df.y, z=df.z, text=df.file_name, 
                                customdata=[df.path, df.file_name], 
                                mode='markers')])
    scatter = f.data
    f.layout.hovermode = 'closest'
    f.update_traces(hovertemplate="%{text}<extra></extra>") 
    for s in scatter:
        s.on_click(lambda trace, points, selector: play_sound(df.iloc[points.point_inds].file_name.item()))
       
        s.marker.colorscale = 'agsunset'
        s.marker.color = df.avg_amp

    f.layout.plot_bgcolor = 'black'
    f.layout.paper_bgcolor = 'black'
    f.layout.font = {'color': 'white'}

    return f
    
@app.callback(
    Output('tsne-output', 'figure'),
    Input('generate-tsne-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('perplexity-slider', 'value'),
    State('learning-rate-slider', 'value'),
    prevent_initial_call=True  # This prevents the callback from running at app startup
)
def generate_tsne(n_clicks, model_id, perplexity, learning_rate):
    if n_clicks == 0:
        return dash.no_update

    # Your t-SNE plot generation logic here
    plot = tsne_plot(model_id, perplexity, learning_rate)
    return plot

# Interpolation code
@app.callback(
    [Output('node1-output', 'children'), Output('node1-audio', 'src')],
    [Input('set-node1-button', 'n_clicks')],
    [State('tsne-output', 'clickData')]
)
def set_node1(n_clicks, clickData):
    if n_clicks == 0 or clickData is None:
        return ('', '')
    # Retrieve the file path from the clicked point
    path = clickData['points'][0]['text']
    
    return (path, '/audio/' + path)

@app.callback(
    [Output('node2-output', 'children'), Output('node2-audio', 'src')],
    [Input('set-node2-button', 'n_clicks')],
    [State('tsne-output', 'clickData')]
)
def set_node2(n_clicks, clickData):
    if n_clicks == 0 or clickData is None:
        return ('', '')
    
    path = clickData['points'][0]['text']

    return (path, '/audio/' + path)

@app.callback(
    Output('audio-container', 'children', allow_duplicate=True),
    [Input('interpolate-button', 'n_clicks')],
    [State('model-dropdown', 'value'), 
     State('node1-output', 'children'), 
     State('node2-output', 'children'), 
     State('steps-slider', 'value'),
     State('audio-container', 'children')
     ]
)
def interpolate(n_clicks, model_id, node1, node2, steps, audio_children):
    if n_clicks == 0 or node1 == '' or node2 == '':
        return []



    pickle_file = f"{model_id.replace('/', '-')}.pkl"
    pipe = None 

    if os.path.exists(pickle_file):
        print(f"Loading {model_id} from pickle file")
        with open(pickle_file, 'rb') as f:
            pipe = pickle.load(f)
        print(f"Loaded {model_id} from pickle file")
    else:
        print(f"Downloading {model_id} from HuggingFace")
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        print(f"Saving {model_id} to pickle file")
        with open(pickle_file, 'wb') as f:
            pickle.dump(pipe, f)
    
    sr = pipe.unet.sample_rate

    audio1 = librosa.load("output/" + node1, sr=sr)
    audio2 = librosa.load("output/" + node2, sr=sr)

    audios = [pipe.encode([audio1]), pipe.encode([audio2])]

    interpolated_audios = pipe.slerp(audios[0], audios[1], steps=steps, alpha=0.5)

    audio_elements = []
    for i, audio in enumerate(interpolated_audios):
        filename = f"interpolated_{model_id.replace('/', '-')}-{length_in_s}s-{steps}-{index}.wav"
        write(filename, sr, audio) 
        audio_elements.append(html.Audio(id=f'audio-player-{i}', controls=True, autoPlay=True, src=f"/audio/{filename}"))

    return audio_elements

@app.callback(
    Output('audio-container', 'children'),
    Input('generate-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('steps-slider', 'value'),
    State('length-slider', 'value'),
    State('batch-size-slider', 'value'),
    prevent_initial_call=True  # This prevents the callback from running at app startup
)
def generate(n_clicks, model_id, steps, length_in_s, batch_size):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update

    pickle_file = f"pipes/{model_id.replace('/', '-')}.pkl"
    pipe = None

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            pipe = pickle.load(f)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        with open(pickle_file, 'wb') as f:
            pickle.dump(pipe, f)

    pipe = pipe.to(device)

    try:
        audios = pipe(audio_length_in_s=length_in_s, num_inference_steps=steps, batch_size=batch_size).audios
    except AttributeError:
        audios = pipe(num_inference_steps=steps, batch_size=batch_size).audios

    files = os.listdir('output')
    matching_files = [f for f in files if f.startswith(f"{model_id.replace('/', '-')}-{length_in_s}s-{steps}")]
    max_index = len(matching_files)

    audio_elements = []
    for i, audio in enumerate(audios):
        index = max_index + i
        filename = f"{model_id.replace('/', '-')}-{length_in_s}s-{steps}-{index}.wav"
        write(f"output/{filename}", pipe.unet.sample_rate, audio.transpose())
        audio_elements.append(html.Audio(id=f'audio-player-{i}', controls=True, autoPlay=False, src=f"/audio/{filename}"))

    return audio_elements

# Model definitions
models = [
    "harmonai/glitch-440k",
    "harmonai/jmann-small-190k",
    "harmonai/jmann-large-580k",
    "harmonai/maestro-150k",
    "harmonai/unlocked-250k",
    "harmonai/honk-140k"
]

perplexity_slider = dcc.Slider(
    id='perplexity-slider',
    min=5,
    max=50,
    step=5,
    value=30,  # Default value
    marks={i: str(i) for i in range(5, 51, 5)},
    tooltip={"placement": "bottom", "always_visible": True}
)

learning_rate_slider = dcc.Slider(
    id='learning-rate-slider',
    min=10,
    max=1000,
    step=10,
    value=200,  # Default value
    marks={i: str(i) for i in range(10, 1001, 100)},
    tooltip={"placement": "bottom", "always_visible": True}
)

# App code
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': i, 'value': i} for i in models],
            value="harmonai/glitch-440k"
        ),
        dcc.Slider(
            id='batch-size-slider',
            min=1,
            max=8,
            step=1,
            value=1,
        ),
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        dcc.Slider(
            id='steps-slider',
            min=1,
            max=500,
            step=50,
            value=200,
        ),
        dcc.Slider(
            id='length-slider',
            min=1,
            max=30,
            step=1,
            value=2,
        ),
    ], style={'width': '50%', 'display': 'inline-block'}),

    # New Div for Perplexity and Learning Rate Sliders
    html.Div([
        html.Label('Perplexity:'),
        perplexity_slider,
        html.Label('Learning Rate:'),
        learning_rate_slider
    ], style={'position': 'absolute', 'left': '10px', 'bottom': '10px', 'width': '300px'}),

    # Existing layout components continue here...
    html.Div([
        html.Div([
            html.Audio(id='audio-player', controls=True, autoPlay=True, title='Clicked node audio'),            
            html.Div([
                dcc.Loading(
                    id="loading-audio",
                    type="default",
                    children=html.Div(id='audio-container', children=[html.Audio(controls=True)], style={'display': 'flex', 'flex-direction': 'column'}, title="Generated audio")
                )
            ], style={'display': 'inline-block'}),

        ], style={'display': 'inline-block'}),
        html.Div([
            html.Div([
            html.Div([
                html.Button('Generate Audio', id='generate-button', n_clicks=0),
                html.Button('Generate t-SNE Plot', id='generate-tsne-button', n_clicks=0),            
            ]),
            html.Div([
                html.Button('Set Node 1', id='set-node1-button', n_clicks=0),
                html.Button('Set Node 2', id='set-node2-button', n_clicks=0),
                html.Button('Interpolate', id='interpolate-button', n_clicks=0),                
            ]),
            html.Audio(id='node1-audio', controls=True),
            html.Audio(id='node2-audio', controls=True),

            ], style={'display': 'flex'}),
            html.Div([
                dcc.Loading(
                    id="loading-tsne",
                    type="default",
                    children=dcc.Graph(id='tsne-output', config={'displayModeBar': False, 'autosizable': True, 'fillFrame': True}, style={'width': '100%', 'height': '100%'}, figure=placeholder_plot)
                )
            ], style={'width': '100%', 'height': '100%'}),
        ], style={'width': '100%', 'height': '100%'}),  
                
    ], style={'width': '100%', 'height': '100%', 'display': 'flex'}),
    html.Div(id='node1-output', style={'display': 'none'}),
    html.Div(id='node2-output', style={'display': 'none'}),
    html.Div(id='model_sample_rate', style={'display': 'none'}),    

])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')