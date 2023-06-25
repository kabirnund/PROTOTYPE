
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
import spotipy
import seaborn as sns
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_option_menu import option_menu
import hashlib
import datetime
import os
import io
import smtplib
from email.message import EmailMessage




from yellowbrick.cluster import KElbowVisualizer
import streamlit.components.v1 as components
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

#import polarplot
import matplotlib



import plotly.offline as pyo
import plotly.graph_objs as go
#from matplotlib import polarplot

import plotly.graph_objects as go
#it_tags import st_tags, st_tags_sidebar
astro_animation_html = """
<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
<lottie-player src="https://assets4.lottiefiles.com/packages/lf20_euaveaxu.json"  background="transparent"  speed="1"  style="width: 200px; height: 180px;"  loop  autoplay></lottie-player> """


music_animation_html = """

<lottie-player src="https://assets4.lottiefiles.com/packages/lf20_euaveaxu.json"  background="transparent"  speed="1"  style="width: 170px; height: 160px;"  loop  autoplay></lottie-player> """

if 'login_success' not in st.session_state:
    st.session_state.login_success = False
if 'username_loggedin' not in st.session_state:
    st.session_state.username_loggedin = ''
if 'results' not in st.session_state:
    st.session_state.results = ''




def login_user(username, password):
    a = pd.read_csv("new.csv")
    b = a.loc[(a['Username'] == username) & (a['Password'] == password)]
    print(b)
    c = b.to_numpy()
    return c

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
cols = ['Username', 'Password', 'Count', 'Timestamp', 'loved_it', 'like_it', 'okay', 'hate_it', 'recently_searched_song', 'email_id', 'rec_song_uri']
lst = []
def add_userdata(username,password,count,time, loved_it, like_it, okay, hate_it, recently_searched_song, email_id, rec_song_uri):
    lst.append([username,password,count,time, loved_it, like_it, okay, hate_it, recently_searched_song, email_id, rec_song_uri])
    path = 'C:/Users/skylo/OneDrive/Desktop/new.csv'

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        df = pd.DataFrame(lst, columns = cols)
        df.to_csv("new.csv", index = False)
    else:
        df = pd.read_csv("new.csv")
        df = df.append(pd.DataFrame(lst, columns = cols), ignore_index=True)
        df.to_csv("new.csv", index = False)
    print(df)
    return df

def view_all_users():
    # Your implementation for retrieving all users from the database goes here
    pass



def local_html(file_name):
    with open(file_name, 'r') as f:
        page = f.read()
    return page




a='378d3fd44c3e4c01a2288400de3d109d'
b='1d0556c14b4740d7a396531ad76ed0f7'
# Set up the client credentials manager
client_id = a
client_secret = b
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Create a Spotify client using the client credentials manager
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@st.cache_resource
def load_data():
    #track_origin = pd.read_csv('tracks.csv')
    #track=track_origin.iloc[:100]
    dataset = pd.read_csv('data.csv')
    #dataset=dataset_origin.iloc[:100]
    #dataset.to_csv('data_subset.csv', index=False)
    #genreby = pd.read_csv('data_by_genres.csv')
    #yearby = pd.read_csv('data_by_year.csv')
    return dataset
    #return track
    #return genreby
    #return yearby
def show_polar_plot(df):
                                    fig = go.Figure(
                                        go.Scatterpolar(
                                        r=df.values[0],
                                        theta=df.columns,
                                        mode='lines+markers',
                                        line_color='blue',
                                        marker=dict(
                                            color='blue',
                                            symbol='circle'
                                        )
                                    )
                                )
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                                ),
                                            angularaxis=dict(
                                                visible=True,
                                                direction='clockwise'
                                                )
                                            ),
                                        showlegend=False
                                        )
                                    st.plotly_chart(fig)


def send_email(song_list,retrieved_email):
    EMAIL_ADDRESS = os.environ.get('lamizikmusicrecommender@gmail.com')
    EMAIL_PASSWORD = os.environ.get('qfqpzrizgpoegxcz')

    

    msg = EmailMessage()
    msg['Subject'] = 'Check out Bronx as a puppy!'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = retrieved_email

    msg.set_content('This is a plain text email')
    spotify_url = 'https://open.spotify.com/embed/track/0FE9t6xYkqWXU2ahLh6D8X'
    html_table = song_list.to_html(index=False)
    html_content = f"""
    <html>
    <body>
        <h1>Song List</h1>
        {html_table}
    </body>
    </html>
    """
    
    msg.add_alternative(html_content, subtype='html')
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('lamizikmusicrecommender@gmail.com','qfqpzrizgpoegxcz' )
        smtp.send_message(msg)















def input_preprocessor(song_list, dataset):
    features = ['valence', 'year', 'acousticness',
                'danceability', 'duration_ms', 'energy',
                'explicit','instrumentalness', 'key', 
                'liveness', 'loudness', 'mode',
                'popularity','speechiness', 'tempo']
    song_vectors = []
    for song in song_list:
        try:
            song_data = dataset[(dataset['name'] == song['name']) & 
                                (dataset['year'] == song['year'])].iloc[0]
        except IndexError:
            song_data = None
        if song_data is None:
            print('Warning: {} does not exist in our database'.format(song['name']))
            continue
        song_vectors.append(song_data[features].values)  
    return np.mean(np.array(list(song_vectors)), axis=0)

@st.cache_resource
def k_means_cluster(X, k, max_iter=100):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, size=k, replace=False), :]
    for i in range(max_iter):
        distances = np.sqrt(np.sum(np.square(X[:, np.newaxis] - centroids), axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    return centroids, labels

def Music_Recommender(song_list, dataset, n_songs=20):
    features = ['valence', 'year', 'acousticness',
                'danceability', 'duration_ms', 'energy',
                'explicit','instrumentalness', 'key', 
                'liveness', 'loudness', 'mode',
                'popularity','speechiness', 'tempo']
    metadata_cols = ['year', 'name',  'artists']
    X = dataset[features].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    centroids, labels = k_means_cluster(X, k=8)
    song_center = input_preprocessor(song_list, dataset)
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = np.sqrt(np.sum(np.square(X - scaled_song_center), axis=1))
    index = np.argsort(distances)[:n_songs]
    rec_output = dataset.iloc[index]
    # Add links to open.spotify for each song
    rec_output['spotify_link'] = rec_output.apply(lambda row: 
                            f"https://open.spotify.com/embed/track/{row['id']}", axis=1)
    rec_output['spotify_player'] = rec_output.apply(lambda row: 
                        f'<iframe src="https://open.spotify.com/embed/track/{row["id"]}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', axis=1)
    return rec_output[metadata_cols + ['spotify_player']]


def add_spotify_links(rec_output):
    metadata_cols = ['name', 'artists', 'id','year']  # List of metadata columns

    # Add links to open.spotify for each song
    rec_output['spotify_link'] = rec_output.apply(lambda row: 
                            f"https://open.spotify.com/embed/track/{row['id']}", axis=1)
    rec_output['spotify_player'] = rec_output.apply(lambda row: 
                        f'<iframe src="https://open.spotify.com/embed/track/{row["id"]}" width="300" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', axis=1)
    
    return rec_output[metadata_cols + ['spotify_player']]









def main():
    
    st.set_page_config(page_title='Lamizik Music Recommender', page_icon=':notes:', layout='wide')
    if st.session_state.login_success:
            st.success("Welcome  "+st.session_state.username_loggedin+" !")
            a = pd.read_csv("new.csv")
            b = a.loc[(a['Username'] == st.session_state.username_loggedin) ]
            retrieved_email=b.at[b.index[b['Username'] == st.session_state.username_loggedin].tolist()[0],'email_id']
            #st.write(b.at[b.index[b['Username'] == st.session_state.username_loggedin].tolist()[0],'email_id'])
    #cols=st.columns(2)
    #with cols[0]:
    #st.title('Lamizik Music Recommender')
    #with cols[1]:    
        #st.image('https://www.freepnglogos.com/uploads/spotify-logo-png/spotify-download-logo-30.png', width=100)
                        #left_col, right_col = st.columns([1, 7])
                        #with left_col:
                        #    components.html(astro_animation_html)
                        #with right_col:
                           # st.markdown("<p style = 'font-size: 36px; font-weight: bold;'> <br> Music Recommender System ..</p>""",unsafe_allow_html = True)
    #menu = ['Recommender', 'EDA','K-means cluster']
    #choice = st.sidebar.selectbox('Select page', menu)
    # Hero section
    
    
    
    
    
    
    with st.sidebar:
    #model_button = st.button('Recommendation')
    #print(os.getcwd())
        #print('./assets/images/spotify.png')
        #st.image('./assets/images/spotify.png')
        choose = option_menu("Lamizik Music Recommender", ["Home", "EDA", "K-means cluster", "Recommender", "Your Profile"],
                                icons=['house', 'file-earmark-music-fill', 'pc', 'boombox','journal'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#000000", "font": "proxima nova"},
                "icon": {"color": "#1DB954", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#000111"},
                "nav-link-selected": {"background-color": "#181818"},
            }
            )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    dataset = load_data()
    if 'selected_df' not in st.session_state:
        st.session_state.selected_df = pd.DataFrame()
    
    
    if choose == 'Home':
        st.header('Home')
       
        
        st.markdown(
        """
        <div style="background-color: #1DB954; padding: 2rem; text-align: center;">
            <h1 style="color: white; font-size: 3rem;">Lamizik Music Recommender System</h1>
            <p style="color: white; font-size: 1.5rem;">Discover and explore your favorite tunes.</p>
        </div>
        """,
            unsafe_allow_html=True
            )
        
        #with st.container():
            #left_col, right_col = st.columns([2, 2])
            #with left_col:
                #st.markdown(
        #"""
        #<style>
        #.iframe-container {
            #position: relative;
            #width: 100%;
            #height: 0;
            #padding-bottom: 50%; /* 16:9 aspect ratio */
            #margin-top: 2rem;
            #overflow: hidden;
        #}
        
        #.iframe-container iframe {
            #position: absolute;
            #top: 0;
            #left: 0;
            #width: 100%;
            #height: 50%;
            #border: none;
        #}
        #</style>
        
        #<div class="iframe-container">
            #<iframe src="https://embed.lottiefiles.com/animation/51847"></iframe>
        #</div>
        #""",
            #unsafe_allow_html=True
            #)
        
       
            #with right_col:
                #st.markdown("<p style = 'font-size: 36px; font-weight: bold;'> <br> Sit back and listen to recommendations ................................</p>""",unsafe_allow_html = True)
        with st.container():
            
            st.subheader("Login")

            username = st.text_input("User Name")
            password = st.text_input("User Password", type='password')

            if st.button("Login"):
                hashed_pswd = make_hashes(password)
                result = login_user(username, check_hashes(password, hashed_pswd))

                if len(result):
                
                    st.success("Logged In as {}".format(username))
                    st.session_state.login_success = True
                    st.session_state.username_loggedin = username
                    #st.session_state.usermail_loggedin = 
        
        # Add your logic for the authenticated user interface here
        
                else:
                    st.warning("Incorrect Username/Password")

            elif st.checkbox("New User?"):
                
                
                st.subheader("Create New Account")
                new_user = st.text_input("Username")
                new_password = st.text_input("Password", type='password')
                new_email_id = st.text_input("Email ID")

                if st.button("Signup"):
                    add_userdata(new_user,make_hashes(new_password), 0, datetime.datetime.now(), 0, 0, 0, 0, "", new_email_id, '')
                    st.success("You have successfully created a valid Account")
                    st.info("Go to Login Menu to login")                
    
           
    
    
    
    
    
    
    
    
        
    if choose == 'Recommender':
        if st.session_state.login_success:
            
            
            
        
     
            with st.container():
                st.subheader('Input song details:')
            
                song_name = st.text_input('Song Name')
                if not song_name:
                    st.error('Please input a song name')
                else:
                    cols = st.columns(2)
                        
                    with cols[0]:
                        
                
                        songs_filtered = dataset[dataset['name'].str.contains(song_name, case=False)]
                        search_results = songs_filtered[['name', 'artists', 'year','id']]
                
                        st.subheader('Search info:')
                        gb = GridOptionsBuilder.from_dataframe(search_results)
                        gb.configure_selection('single', use_checkbox=True)
                        gridOptions = gb.build()
                        grid_response = AgGrid(search_results, gridOptions=gridOptions, height=200, width='70%')
                        selected_rows = grid_response['selected_rows']
                
                        selected_df = pd.DataFrame(selected_rows)
                                
                    
                        
                        
                    
                    
                    
                        st.subheader('Song Preview:')
                        if not selected_df.empty:
                            search_info=add_spotify_links(selected_df)
                                
                    
                            for i, row in search_info[['name', 'artists', 'year', 'spotify_player']].iterrows():
                                st.components.v1.html(row['spotify_player'], height=380,width=700)               
                
                            st.subheader('Recommendations:')
                            
                       
            
                    with cols[1]:
                        if not selected_df.empty:
                            track_features  = sp.audio_features(selected_df['id'])
                            df = pd.DataFrame(track_features, index=[0])
                            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
                            show_polar_plot(df_features)
                            cols = st.columns([1,2])
                            with st.container():
                                with cols[0]:
                                    components.html(astro_animation_html)
                                with cols[1]:
                                    st.markdown("<p style = 'font-size: 36px; font-weight: bold;'> <br> Sit back and listen to recommendations ..</p>""",unsafe_allow_html = True)    
                                    
                            
                    with st.container():
                        
                    
                       
                        if not selected_df.empty:
                            song_list_sanitized = selected_df[['name','year']]
                            song_list_sanitized_to_dict = song_list_sanitized.to_dict('records')
                            results = Music_Recommender(song_list_sanitized_to_dict, dataset)
                            if st.button('Send Recommendations via mail'):
                                        send_email(results,retrieved_email)
                                        st.success('Email sent successfully to  ' +' '+ retrieved_email)
                            #st.write(results)
                        
                            cols = st.columns(4)
                            for i in range(0,4):
                                with cols[i]: 
                                    
                                    st.components.v1.html(results['spotify_player'].iloc[i], height=380,width=700)
                
                            for i in range(0,4):
                                with cols[i]: 
                                    
                                    st.components.v1.html(results['spotify_player'].iloc[i+4], height=380,width=700)
                            for i in range(0,4):
                                with cols[i]: 
                                    
                                    st.components.v1.html(results['spotify_player'].iloc[i+8], height=380,width=700)                     
                            for i in range(0,4):
                                with cols[i]: 
                                    #st.write(results['spotify_player'].iloc[i])
                                    st.components.v1.html(results['spotify_player'].iloc[i+12], height=380,width=700)
                            for i in range(0,4):
                                with cols[i]: 
                                    #st.write(results['spotify_player'].iloc[i])
                                    st.components.v1.html(results['spotify_player'].iloc[i+16], height=380,width=700)        
                            for i in range(0,4):
                                with cols[i]: 
                                    #st.write(results['spotify_player'].iloc[i])
                                    st.components.v1.html(results['spotify_player'].iloc[i+1], height=380,width=700)
        else:
            st.warning("Please Login to get Recommendations")            
        
                   
    if choose=='K-means cluster':
        st.subheader('K-means cluster')
        st.write('K-means cluster')
        def load_data_k(file_path):
            dataset = pd.read_csv(file_path)
            df = dataset.select_dtypes(np.number)
            return df
        def standardize_data(df):
            scaler = StandardScaler()
            scaled_X = scaler.fit_transform(df.values)
            scaled_df = pd.DataFrame(scaled_X, columns=df.columns)
            return scaled_df
        def perform_pca(scaled_df, n_components):
            pca = PCA(n_components=n_components)
            pca.fit(scaled_df)
            pca_df = pca.transform(scaled_df)
            return pca_df
        def find_optimal_clusters(pca_df):
            inertias = []
            for i in range(2, 11):
                centroids = pca_df[np.random.choice(len(pca_df), i)]
                for _ in range(10):
                    distances = np.linalg.norm(pca_df[:, np.newaxis, :] - centroids, axis=2)
                    labels = np.argmin(distances, axis=1)
                    new_centroids = np.array([pca_df[labels == j].mean(axis=0) for j in range(i)])
                    if np.allclose(new_centroids, centroids):
                        break
                    centroids = new_centroids
                inertias.append(np.sum(np.min(distances, axis=1)))
            return inertias
        def perform_kmeans(pca_df, num_clusters):
            centroids = pca_df[np.random.choice(len(pca_df), num_clusters)]
            for _ in range(10):
                distances = np.linalg.norm(pca_df[:, np.newaxis, :] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([pca_df[labels == j].mean(axis=0) for j in range(num_clusters)])
                if np.allclose(new_centroids, centroids):
                    break
                centroids = new_centroids
                return labels, centroids
            
     
        def create_scatter_plot(pca_df, labels, centroids):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter3D(pca_df[:, 0], pca_df[:, 1], pca_df[:, 2], c=labels, cmap='rainbow')
            ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=300, linewidths=3, color='black')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
            ax.set_title('KMeans Clustering')
            return fig
        
        df = load_data_k("data.csv")
        scaled_df = standardize_data(df)
        pca_df = perform_pca(scaled_df, n_components=3)
        inertias = find_optimal_clusters(pca_df)
        plt.plot(range(2, 11), inertias, '-o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.xticks(range(2, 11))
        st.pyplot(plt)
        #num_clusters = 8
        num_clusters = st.selectbox('Select the number of clusters', range(2, 11))
        labels, centroids = perform_kmeans(pca_df, num_clusters)
        fig=create_scatter_plot(pca_df, labels, centroids)
        st.pyplot(fig)
        centroid_data = pd.DataFrame(centroids, columns=['Centroid X', 'Centroid Y', 'Centroid Z'])
        st.subheader('Centroid Coordinates')
        st.write(centroid_data)                
    
    if choose == 'EDA':
        
        st.write('EDA')
        st.header("Exploratory Data Analysis(EDA)")
        st.markdown("Simply defined, exploratory data analysis (EDA for short) is looking for patterns and summarizing the dataset’s main characteristics beyond what they learn from modeling and hypothesis testing. EDA is a philosophy that allows us to approach a database without assumptions. When a data analyst employs EDA, it’s like they’re asking the data to tell them what they don’t know. ")
        track1=pd.read_csv('tracks.csv')
        dataset1 = pd.read_csv('data.csv')
        genreby1 = pd.read_csv('data_by_genres.csv')
        yearby1 = pd.read_csv('data_by_year.csv')
        st.subheader('1. Count plot')


       # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

       # Add suptitle
        fig.suptitle('count plots')

       # Plot count plots
        sns.countplot(ax=axes[0], x='explicit', data=track1, palette='coolwarm')
        sns.countplot(ax=axes[1], x='mode', data=track1, palette='coolwarm')

       # Render the plots using Streamlit
        st.pyplot(fig)
        st.markdown('- from the count plots it is very clear that the number of explict songs are quiet few')
        st.markdown('- and for the mode feature classification the major is almost double of minor ')
        st.subheader('2.Various features & their change in trend with time')
        zerotoone = yearby1[['acousticness', 'danceability',
                    'energy', 'instrumentalness',
                    'liveness', 'valence','year']]
        df = zerotoone.copy()
        dfn = df[df.year > 1945]
        dfn = dfn.reset_index(drop=True)
        dfn.set_index('year', inplace=True)

       # Set Seaborn style
       #sns.set_style('whitegrid')

       # Plot the dataframe using line plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.lineplot(data=dfn, ax=ax)

       # Remove spines 
        sns.despine(left=True, bottom=True)
       

       # Render the plot using Streamlit
        st.pyplot(fig)
        st.markdown('- Acousticness of songs decreased with time, there was a very sharp drop 1965 to 1975')
        st.markdown('- Dancebility of songs increased with time')
        st.markdown('- there was a rapid increase in the Energy of the songs in th3 60s')
        st.markdown('- Instrumentalness decreased with the increase in time')
        st.markdown('- Liveness and Valence have remained quite stable throughout the time')
       
        st.subheader('3. Popularity VS Time')
        fig = sns.histplot(x='popularity', data=track1, color="olive")
        st.pyplot(fig.figure)
        st.markdown('- Plot one represents the histogram of the popularity column')
        st.markdown('- plot two is a scatterplot of the songs with popularity greater than 85 that is then plotted against the year of release')
       
       
       
        st.subheader('4. Correlation of Features ')

        fig = plt.figure(figsize=(10,6))

        corr = track1.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, vmin=-1, cmap='viridis', annot=False)

        st.pyplot(fig)
        st.markdown('- energy, loudness and year are highly coreleated positively with popularity')
        st.markdown('- acousticness is inversely corelated with popularity')
       
       
        pop_corr = corr[abs(corr['popularity']) > 0.25]['popularity']

        st.write(pop_corr)
        st.subheader('4. Dancebility ')
        fig = plt.figure(figsize=(10,8))
        sns.set_style('whitegrid')

        sns.scatterplot(x="danceability", y="popularity",hue="year", size="key",
                palette="ch:r=-.4,d=.2_r",
                sizes=(50, 300), linewidth=0,
                data=yearby1,legend=True).set(title='danceability')

        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        st.subheader('4. Artists Popularity ')
        post = dataset1[dataset1['artists'].str.contains('Post Malone')]
        ed = dataset1[dataset1['artists'].str.contains('Ed Sheeran')]
        kw = dataset1[dataset1['artists'].str.contains('Kanye West')]
        dra = dataset1[dataset1['artists'].str.contains('Drake')]
        cb = dataset1[dataset1['artists'].str.contains('Chris Brown')]     
       # load data into `post`, `ed`, `kw`, `dra`, and `cb` variables

        fig, ax = plt.subplots(figsize=(20,10))
        sns.despine(fig, left=True, bottom=True)
        sns.set_context("talk",font_scale=1, rc={"lines.linewidth": 2.5})

        sns.histplot(post['popularity'], color='y',label="Post Malone]")
        sns.histplot(ed['popularity'], color='b',label="Ed Sheeran")
        sns.histplot(kw['popularity'], color='m',label="Kanye West")
        sns.histplot(dra['popularity'], color='g',label="Drake")
        sns.histplot(cb['popularity'], color='r',label="Chris brown")

        ax.legend(fontsize=14)
        st.pyplot(fig)
        st.markdown('i have selected 5 random aritsts:')
        st.markdown('- Post Malone')
        st.markdown('- Ed Sheeran')
        st.markdown('- Kanye West')
        st.markdown('- Drake')
        st.markdown('- Chris Brown')
        st.markdown("**Drake has the highest popularity from these 5 artists**")
       
       
       
        st.subheader('4. Point Chart for various features against Popularity ')
        ld = dataset1.groupby("loudness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()
        ac = dataset1.groupby("acousticness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()
        en = dataset1.groupby("energy")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()
        ins = dataset1.groupby("instrumentalness")["popularity"].mean().sort_values(ascending=False).head(20).reset_index()

        fig, axes = plt.subplots(ncols=2, figsize=(24,6))

        sns.pointplot(data=ld,x="loudness", y='popularity', ax=axes[0])
        axes[0].set_xlabel('loudness', fontsize=12)
        axes[0].set_ylabel('popularity', fontsize=12)
        axes[0].set_title("popularity wise loudness", fontsize=15)
        axes[0].tick_params(axis='x', labelrotation=90)

        sns.pointplot(data=en,x="energy", y='popularity',color='g', ax=axes[1])
        axes[1].set_xlabel('energy', fontsize=12)
        axes[1].set_ylabel('popularity', fontsize=12)
        axes[1].set_title("popularity wise energy", fontsize=15)
        axes[1].tick_params(axis='x', labelrotation=90)

        sns.despine(fig, left=True, bottom=True)
        st.pyplot(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig1, ax1 = plt.subplots(figsize=(12,6))
        sns.pointplot(data=ac,x="acousticness", y='popularity',color='y', ax=ax1)
        ax1.set_xlabel('acousticness', fontsize=12)
        ax1.set_ylabel('popularity', fontsize=12)
        ax1.set_title("popularity wise acousticness", fontsize=15)
        ax1.tick_params(axis='x', labelrotation=90)

        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(12,6))
        sns.pointplot(data=ins,x="instrumentalness", y='popularity',color='r', ax=ax2)
        ax2.set_xlabel('instrumentalness', fontsize=12)
        ax2.set_ylabel('popularity', fontsize=12)
        ax2.set_title("popularity wise instrumentalness", fontsize=15)
        ax2.tick_params(axis='x', labelrotation=90)

        st.pyplot(fig2)                
    
    if choose=="Your Profile":
        st.write("Your Profile")
        if st.session_state.login_success:
            st.write("You are logged in")
            st.header("Your Profile Details")
            st.write("Name: ",st.session_state.username_loggedin)
            st.write("Email: ",retrieved_email)
        else:
            st.warning("Please Login to continue")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                        
                               
                    
                                            
                
                
                
              
if __name__ == "__main__":
    main()           
                    
                           