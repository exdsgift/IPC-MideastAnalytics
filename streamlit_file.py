import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium 
from wordcloud import WordCloud


fatalities_df = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')

st.title('Exploring Terrorism Victim Data: A Look into the Israeli-Palestinian Conflict')
st.subheader('Programming and Database Course Final Project')
st.write('Author: Gabriele Durante')