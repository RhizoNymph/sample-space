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
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.cluster import KMeans
from scipy.io.wavfile import write
import librosa
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np


# Init code
device = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_DIR = os.getenv('AUDIO_DIR', './audio')
upload_dir = os.path.join(AUDIO_DIR, 'uploaded')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    
generator = torch.Generator(device=device)

placeholder_plot = go.Figure()
placeholder_plot.layout.plot_bgcolor = 'black'
placeholder_plot.layout.paper_bgcolor = 'black'
placeholder_plot.layout.font = {'color': 'white'}

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY], prevent_initial_callbacks="initial_duplicate")

@app.callback(
    [Output('audio-player', 'src'),
    Output('path-to-copy', 'children')],
    [Input('tsne-output', 'clickData')]
)
def play_sound(clickData):
    if clickData is None:
        return '', ''
    full_path = clickData['points'][0]['text']
    relative_path = os.path.relpath(full_path, AUDIO_DIR)
    return f'/audio/{relative_path}', relative_path

@server.route('/audio/<path:path>')
def serve_audio(path):
    return send_from_directory(AUDIO_DIR, path)

def is_audio_file(filename):
    if filename.endswith('.wav'):
        return True
    if filename.endswith('.mp3'):
        return True
    if filename.endswith('.ogg'):
        return True
    return False

def get_features(y, sr):
    # Harmonic and Percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Mel-scaled power spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    
    # Chroma feature
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # MFCC
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    
    # Temporal features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Tempo and Beat Tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    rms_energy = librosa.feature.rms(y=y)[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    poly_features = librosa.feature.poly_features(y=y, sr=sr)[0]
    
    # Concatenate all features
    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1), np.mean(delta_mfcc, axis=1), np.mean(delta2_mfcc, axis=1),
        np.mean(chroma, axis=1), np.mean(contrast, axis=1),
        [np.mean(zero_crossing_rate), tempo, len(beat_times), np.mean(rms_energy), np.mean(spectral_rolloff),
         np.mean(spectral_bandwidth), np.mean(spectral_flatness), np.mean(poly_features)]
    ])
    
    
    scaler = StandardScaler()
    feature_vector = scaler.fit_transform(feature_vector.reshape(-1, 1)).flatten()

    return feature_vector, np.mean(y)


from MulticoreTSNE import MulticoreTSNE as TSNE

def tsne_plot(perplexity=30, learning_rate=200, n_iter=1000, n_clusters=5, audio_dir='./audio'):
    files = glob.glob(os.path.join(audio_dir, '**/*'), recursive=True)
    files = [entry for entry in files if os.path.isfile(entry) and is_audio_file(entry)]

    feature_vectors = []
    for f in files:
        y, sr = librosa.load(f)
        feat, avg_amp = get_features(y, sr)
        feature_vectors.append(feat)

    feature_vectors = np.array(feature_vectors)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_vectors)
    labels = kmeans.labels_

    tsne = TSNE(n_components=3, learning_rate=learning_rate, perplexity=perplexity,
                n_iter=n_iter, verbose=0, angle=0.1, random_state=0, n_jobs=-1)  # n_jobs=-1 uses all available cores
    tsne_results = tsne.fit_transform(feature_vectors)

    data = []
    for i, f in enumerate(files):
        abspath = os.path.abspath(f)
        file_name = os.path.basename(f)
        data.append([abspath, tsne_results[i, 0], tsne_results[i, 1], tsne_results[i, 2], labels[i]])

    df = pd.DataFrame(data, columns=['file_name', 'x', 'y', 'z', 'cluster'])

    f = go.FigureWidget([go.Scatter3d(
        x=df.x, y=df.y, z=df.z,
        text=df.file_name,
        mode='markers',
        marker=dict(
            size=5,
            color=df.cluster,
            colorscale='Viridis',
            colorbar=dict(title='Cluster')
        )
    )])

    f.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='closest',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font={'color': 'white'}
    )

    return f
    
@app.callback(
    Output('tsne-output', 'figure'),
    [Input('generate-tsne-button', 'n_clicks')],
    [State('perplexity-slider', 'value'),
     State('learning-rate-slider', 'value'),
     State('iterations-slider', 'value'),
     State('clusters-slider', 'value'),
     State('audio-dir-input', 'value')],  # Add this line
    prevent_initial_call=True
)
def generate_tsne(n_clicks, perplexity, learning_rate, n_iter, n_clusters, audio_dir):  # Add audio_dir parameter
    if n_clicks == 0:
        return dash.no_update

    plot = tsne_plot(perplexity, learning_rate, n_iter, n_clusters, audio_dir)  # Pass audio_dir to tsne_plot
    return plot

clusters_slider = dcc.Slider(
    id='clusters-slider',
    min=2,
    max=10,
    step=1,
    value=5,  # Default value
    marks={i: str(i) for i in range(2, 21)},
    tooltip={"placement": "bottom", "always_visible": True}
)

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
    min=100,
    max=1000,
    step=100,
    value=100,  # Default value
    marks={i: str(i) for i in range(100, 1001, 100)},
    tooltip={"placement": "bottom", "always_visible": True}
)

iterations_slider = dcc.Slider(
    id='iterations-slider',
    min=1000,
    max=5000,
    step=500,
    value=1000,  # Default value
    marks={i: str(i) for i in range(1000, 5001, 1000)},
    tooltip={"placement": "bottom", "always_visible": True}
)

steps_slider = dcc.Slider(
    id='steps-slider',
    min=0,
    max=500,
    step=10,
    value=100,  # Default value
    marks={i: str(i) for i in range(0, 501, 100)},
    tooltip={"placement": "bottom", "always_visible": True}
)

seconds_start_slider = dcc.Slider(
    id='seconds-start-slider',
    min=0,
    max=45,
    step=1,
    value=0,  # Default value
    marks={i: str(i) for i in range(0, 46, 5)},
    tooltip={"placement": "bottom", "always_visible": True}
)

seconds_total_slider = dcc.Slider(
    id='seconds-total-slider',
    min=0,
    max=45,
    step=1,
    value=45,  # Default value
    marks={i: str(i) for i in range(0, 46, 5)},
    tooltip={"placement": "bottom", "always_visible": True}
)

cfg_slider = dcc.Slider(
    id='cfg-slider',
    min=0,
    max=25,
    step=1,
    value=0,  # Default value
    marks={i: str(i) for i in range(0, 26, 1)},
    tooltip={"placement": "bottom", "always_visible": True}
)

positive_prompt_textarea = dcc.Textarea(
    id='positive-prompt-textarea',
    placeholder='Enter a positive prompt...'
    
)

negative_prompt_textarea = dcc.Textarea(
    id='negative-prompt-textarea',
    placeholder='Enter a negative prompt...'
    
)

# Adjustments to the app layout for toolbar, sidebar, and audio player

# Update the app.layout section
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('Positive Prompt:'),
            positive_prompt_textarea,
            html.Label('Negative Prompt:'),
            negative_prompt_textarea,
            html.Button('Generate Audio', id='generate-audio-button', n_clicks=0),
            html.Label('Steps:'),
            steps_slider,
            html.Label('Seconds Start:'),
            seconds_start_slider,
            html.Label('Seconds Total:'),
            seconds_total_slider,
            html.Label('CFG:'),
            cfg_slider,
            html.Label('Audio Directory:'),
            dcc.Input(id='audio-dir-input', type='text', value='./audio', style={'margin': '10px'}),
            html.Button('Generate t-SNE Plot', id='generate-tsne-button', n_clicks=0),
        ], style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'}),
        html.Div([
            html.Label('Perplexity:'),
            perplexity_slider,
            html.Label('Learning Rate:'),
            learning_rate_slider,
            html.Label('Number of Iterations:'),
            iterations_slider,
            html.Label('Number of Clusters:'),
            clusters_slider,
        ], style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'}),
    ], style={'position': 'absolute', 'left': '10px', 'top': '60px', 'width': '300px', 'padding': '20px'}),

    html.Div([
        html.Audio(id='audio-player', controls=True, autoPlay=True, title='Clicked node audio'),
        html.Div([
            html.Div(id='path-to-copy', style={'display': 'inline-block', 'flex-grow': 1}),
            dcc.Clipboard(target_id='path-to-copy', style={'display': 'inline-block'}),
        ], style={'display': 'flex', 'width': '100%', 'justify-content': 'space-between'}),
    ], style={'position': 'absolute', 'top': '10px', 'left': '10px', 'width': '100%', 'padding': '10px'}),

    html.Div([
        dcc.Loading(
            id="loading-tsne",
            type="default",
            children=dcc.Graph(id='tsne-output', config={'displayModeBar': False, 'autosizable': True, 'fillFrame': True}, style={'width': 'calc(100% - 320px)', 'height': '100%'}, figure=placeholder_plot)
        )
    ], style={'position': 'absolute', 'left': '320px', 'top': '10px', 'right': '10px', 'bottom': '10px'}),
], style={'width': '100%', 'height': '100%', 'display': 'flex', 'position': 'relative'})

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)