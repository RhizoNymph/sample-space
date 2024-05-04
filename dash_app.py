# dash_app.py
import os
import pickle
import dash_bootstrap_components as dbc
from flask import Flask, send_from_directory
import glob
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
    # Assuming the full path is provided and 'audio/' is part of the path
    full_path = clickData['points'][0]['text']
    # Extract the relative path from the full path
    audio_base_path = 'audio/'  # Adjust this path to where your 'audio/' directory is located
    relative_path = os.path.relpath(full_path, audio_base_path)
    
    return f'/audio/{relative_path}'

@server.route('/audio/<path:path>')
def serve_audio(path):
    return send_from_directory('audio', path)

def is_audio_file(filename):
    if filename.endswith('.wav'):
        return True
    if filename.endswith('.mp3'):
        return True
    if filename.endswith('.ogg'):
        return True
    return False

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

def tsne_plot(perplexity=30, learning_rate=200, n_iter=1000):
    files = glob.glob('audio/**/*', recursive=True)
    files = [entry for entry in files if os.path.isfile(entry) if is_audio_file(entry)]

    feature_vectors = []
    for f in tqdm(files):
        y, sr = librosa.load(f)
        feat, avg_amp = get_features(y, sr)
        feature_vectors.append({"file": f, "features": feat, "avg_amp": avg_amp})

    tsne = TSNE(n_components=3, learning_rate=learning_rate, perplexity=perplexity,
                n_iter=n_iter, verbose=1, angle=0.1, random_state=0)
    tsne = tsne.fit_transform(np.array([f["features"] for f in feature_vectors]))
    data = []

    for i, f in enumerate(feature_vectors):
        abspath = os.path.abspath(f['file'])
        file_name = os.path.basename(f['file'])
        data.append([abspath, tsne[i,0], tsne[i,1], tsne[i,2], f['avg_amp']])

    df = pd.DataFrame(data, columns =['file_name','x','y', 'z', 'avg_amp'])

    df.to_csv('feature_vectors.csv', index=False)

    f = go.FigureWidget([go.Scatter3d(x=df.x, y=df.y, z=df.z, text=df.file_name, 
                                customdata=[df.file_name], 
                                mode='markers')])
    scatter = f.data
    # Adjust margins
    f.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # Adjust left, right, top, bottom margins as needed
        hovermode='closest'
    )

    # Ensure responsiveness
    f.update_layout(
        autosize=True
    )
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
    [Input('generate-tsne-button', 'n_clicks')],
    [State('perplexity-slider', 'value'),
     State('learning-rate-slider', 'value'),
     State('iterations-slider', 'value')],  # Add this line
    prevent_initial_call=True  # This prevents the callback from running at app startup
)
def generate_tsne(n_clicks, perplexity, learning_rate, n_iter):  # Add n_iter parameter
    if n_clicks == 0:
        return dash.no_update

    # Your t-SNE plot generation logic here
    plot = tsne_plot(perplexity, learning_rate, n_iter)  # Pass n_iter to tsne_plot
    return plot

perplexity_slider = dcc.Slider(
    id='perplexity-slider',
    min=5,
    max=50,
    step=5,
    value=5,  # Default value
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

iterations_slider = dcc.Slider(
    id='iterations-slider',
    min=500,
    max=5000,
    step=500,
    value=1000,  # Default value
    marks={i: str(i) for i in range(500, 5001, 1000)},
    tooltip={"placement": "bottom", "always_visible": True}
)

# App code
app.layout = html.Div([
    # New Div for Perplexity, Learning Rate, and Number of Iterations Sliders
    html.Div([
        html.Label('Perplexity:'),
        perplexity_slider,
        html.Label('Learning Rate:'),
        learning_rate_slider,
        html.Label('Number of Iterations:'),
        iterations_slider
    ], style={'position': 'absolute', 'left': '10px', 'bottom': '10px', 'width': '300px'}),

    # Existing layout components continue here...
    html.Div([
        html.Div([
            html.Audio(id='audio-player', controls=True, autoPlay=True, title='Clicked node audio'),            
        ], style={'display': 'inline-block'}),
        html.Div([
            html.Div([
            html.Div([
                html.Button('Generate t-SNE Plot', id='generate-tsne-button', n_clicks=0),            
            ]),
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
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')