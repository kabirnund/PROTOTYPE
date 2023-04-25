
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


from yellowbrick.cluster import KElbowVisualizer
import streamlit.components.v1 as components

#import polarplot
import matplotlib



import plotly.offline as pyo
import plotly.graph_objs as go
#from matplotlib import polarplot

import plotly.graph_objects as go
#it_tags import st_tags, st_tags_sidebar

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
    





def main():
    st.set_page_config(page_title='Music Recommender', page_icon=':notes:', layout='wide')
    st.title('Music Recommender')
    menu = ['Recommender', 'EDA','K-means cluster']
    choice = st.sidebar.selectbox('Select page', menu)
    dataset = load_data()
    if choice == 'Recommender':
        st.write('Input song details:')
        song_name = st.text_input('Song Name')
        # Filter dataset to show only songs that contain the input string
        songs_filtered = dataset[dataset['name'].str.contains(song_name, case=False)]        

        if len(songs_filtered) > 0:
            #song_artist = songs_filtered.iloc[0]['artists']
            #st.write(f"Artist: {song_artist}")
            song_options = songs_filtered['name'].unique()
            song_selected = st.selectbox('Select Song', song_options)
            song_year = songs_filtered[songs_filtered['name']==song_selected]['year'].values[0]
            song_artist1=songs_filtered[songs_filtered['name']==song_selected]['artists'].values[0]
            song_id = songs_filtered[songs_filtered['name']==song_selected]['id'].values[0]
            st.write(f"Song Year: {song_year}")
            st.write(f"Song Artist: {song_artist1}")
            
            st.write(f"Song id: {song_id}")
            
            track_features  = sp.audio_features(song_id)
            df = pd.DataFrame(track_features, index=[0])
            df_features = df.loc[: ,['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
            st.dataframe(df_features)
            with st.expander("See explanation about features description"):
                st.header("Danceability")
                st.markdown("**Danceability** describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.")
                st.header("Energy")
                st.markdown("Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.")
                st.header("Acouticness")
                st.markdown("A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.")
                st.header("Instrumentalness")
                st.markdown("Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.")
                st.header("Liveness")
                st.markdown("Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.")
        
                st.header("Speechiness")
                st.markdown("Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.")
        
                st.header("Valence")
                st.markdown("A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).")
        
                st.header("Tempo")
                st.markdown("The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.")                
        
                st.header("Loudness")
                st.markdown("The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.")
            #polarplot.feature_plot(df_features)
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
            show_polar_plot(df_features)
            
            
            
        else:
            #song_year = st.number_input("Song Year", min_value=1900, max_value=2023, step=1)
            st.write(f"Sorry Song is not found in our database :(")
            
    #if st.button("Recommend"):
        #results = Music_Recommender([{'name': song_name, 'year': song_year}], dataset)
        #st.write("Recommended Songs:")
        ##st.write(table, unsafe_allow_html=True)
       
        
        
        #track = pd.read_csv('tracks.csv')
        #pr = track.profile_report()

        #st_profile_report(pr)
        #st.write('Input song details:')
        #song_name = st_tags_sidebar(label='Song Name', text='', suggestions=list(dataset['name'].unique()))
        #song_year = st.number_input("Song Year", min_value=1900, max_value=2023, step=1)
        #if st.button("Recommend"):
          #results = Music_Recommender([{'name': song_name, 'year': song_year}], dataset)
          #st.write("Recommended Songs:")
        
        
        
        
        
        #track = pd.read_csv('tracks.csv')
        #pr = track.profile_report()

        #st_profile_report(pr)
        #st.write('Input song details:')
        #song_name = st.text_input('Song Name')
        #song_year = st.number_input("Song Year", min_value=1900, max_value=2023, step=1)
        if st.button("Recommend"):
            song_list = [
    {'name': song_selected, 'year': song_year},
    
]
            #results = Music_Recommender([{'name': song_selected, 'year': song_year}], dataset)
            results = Music_Recommender(song_list, dataset)
            with st.expander("Recommended Songs"):
                st.write("Recommended Songs:")
                for i, row in results[['name', 'artists', 'year', 'spotify_player']].iterrows():
                    st.write(f"{i+1}. {row['name']} - {row['artists']} ({row['year']})")
                    st.components.v1.html(row['spotify_player'], height=380,width=700) 
            
            
            
            #st.write(results[['name', 'artists', 'year', 'spotify_link']],unsafe_allow_html=True)  # add unsafe_allow_html parameter
            #table = results[['name', 'artists', 'year', 'spotify_link']].to_html(escape=False, index=False)
            #st.write(table, unsafe_allow_html=True)
            #st.write(rec_output[['name', 'artists', 'spotify_link']], unsafe_allow_html=True)
            #st.dataframe(results[['name', 'artists', 'year', 'spotify_link']])
            #st_player("https://soundcloud.com/imaginedragons/demons")
            #components.iframe("https://open.spotify.com/embed/track/2iuZJX9X9P0GKaE93xcPjk", width=700, height=300)
            #for index, row in results[['name', 'artists', 'year', 'spotify_link']].iterrows():
              #st.write(row['name'], ' - ', row['artists'])
              #st.components.v1.html(f'<iframe src="{row["spotify_link"]}" width="700" height="300" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', height=300)
              
              
            
    

        #song_name = st.text_input('Song Name')
        #song_year = st.number_input("Song Year", min_value=1900, max_value=2023, step=1)
        #if st.button("Recommend"):
            #results = Music_Recommender([{'name': song_name, 'year': song_year}], dataset)
            #st.write("Recommended Songs:")
            #st.write(results[['name', 'artists', 'year', 'spotify_link']],unsafe_allow_html=True)  # add unsafe_allow_html parameter
            #table = results[['name', 'artists', 'year', 'spotify_link']].to_html(escape=False, index=False)
            #st.write(table, unsafe_allow_html=True)
            #st.write(rec_output[['name', 'artists', 'spotify_link']], unsafe_allow_html=True)

           
    elif choice == 'EDA':
        
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

    elif choice == 'K-means cluster':
        
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
        
if __name__ == "__main__":
    main()           
                    