########################################### Packages

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium 
from wordcloud import WordCloud
from io import BytesIO

########################################### Import Dataset

fatalities_df = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')

########################################### Main Titles

st.title('Exploring Terrorism Victim Data: A Look into the Israeli-Palestinian Conflict')
st.subheader('Programming and Database Course Final Project')
st.write('Author: Gabriele Durante')


########################################### Figure
plt.figure(figsize=(10, 8))
plt.xlabel('Date of Death')
plt.ylabel('Age')
plt.title('Age vs. Date of Death (Gender)')
plt.xticks(rotation = 45, fontsize=3)
plt.yticks(rotation = 0, fontsize=3)

sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='gender', palette='cividis')

plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

########################################### From plt to image function

def plt_to_image(plt):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

graph_image = plt_to_image(plt)

st.image(graph_image)

########################################### Figure

plt.figure(figsize=(10, 8))
plt.xlabel('Date of Death')
plt.ylabel('Age')
plt.title('Age vs. Date of Death (Type of Injury)')
plt.xticks(rotation = 45, fontsize=5)
plt.yticks(rotation = 0, fontsize=5)

sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='type_of_injury', palette='rocket')

plt.legend(title='Type of Injury', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

graph_image = plt_to_image(plt)

st.image(graph_image)

########################################### Figure

plt.figure(figsize=(10, 8))
plt.xlabel('Date of Death')
plt.ylabel('Age')
plt.title('Age vs. Date of Death (citizenship)')
plt.xticks(rotation = 45, fontsize=5)
plt.yticks(rotation = 0, fontsize=5)

sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='citizenship', palette='Set2')

plt.legend(title='Citizenship', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

graph_image = plt_to_image(plt)

st.image(graph_image)

########################################### Figure

plt.figure(figsize=(10, 8))
plt.xlabel('Date of Death')
plt.ylabel('Age')
plt.title('Age vs. Date of Death (killed by)')
plt.xticks(rotation = 45, fontsize=5)
plt.yticks(rotation = 0, fontsize=5)

sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age',  hue='killed_by', palette='husl')

plt.legend(title='killed by:', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

graph_image = plt_to_image(plt)

st.image(graph_image)