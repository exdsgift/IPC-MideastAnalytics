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

########################################### Import Dataset and set tabs

fatalities_df = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')
tab_names = ["Introduction", "Cleaning and Correlation", "Exploratory Data Analysis", "Modelling"]
current_tab = st.sidebar.selectbox("Summary", tab_names)

########################################### Main Titles and introduction
if current_tab == "Introduction":
    st.markdown("<h1 style='text-align: center;'>Exploring Terrorism Victim Data: A Look into the Israeli-Palestinian Conflict</h1>", unsafe_allow_html=True)
    st.subheader('Programming and Database Course Final Project')
    st.write('Author: Gabriele Durante')

    st.write('The dataset is available at the following [Kaggle link](https://www.kaggle.com/datasets/willianoliveiragibin/fatalities-in-the-israeli-palestinian). It reports data on victims of terrorism in Israel and the war in Palestine.')
    st.write('''
         This project aims to analyse events and fatalities by

            - Year, month, and day of the month of events that led to fatalities
            - Victim profiles such as age, gender, citizenship, participation in hostilities etc
            - Event locations, location districts, and location regions
            - Type of injury, type of ammunition, and party responsible for killings, among others.
            ''')
########################################### From plt to image function, design def function

    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf

    graph_image = plt_to_image(plt)

########################################### insert containers using with to add graphs
elif current_tab == "Cleaning and Correlation":
    
    st.header("Cleaning and Correlation")
    st.subheader('Check Correlation using scatterplots')
    st.write('A first graphical analysis was carried out using scatterplots as a graphical method, so as to observe how the main variables relate to each other.')
    
    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf

    graph_image = plt_to_image(plt)
    tab1, tab2, tab3, tab4 = st.tabs(["Gender", "Type of Injury", "Citizenship", "Killed by"])

    with tab1:
            plt.figure(figsize=(10, 8))
            plt.xlabel('conflict from 2000 to 2023')
            plt.ylabel('Age')
            plt.title('Age vs. Date of Death (Gender)')
            plt.xticks(rotation = 45, fontsize=3)
            plt.yticks(rotation = 0, fontsize=3)
            sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='gender', palette='cividis')
            plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks([])
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)
    
    with tab2:
            plt.figure(figsize=(10, 8))
            plt.xlabel('conflict from 2000 to 2023')
            plt.ylabel('Age')
            plt.title('Age vs. Date of Death (Type of Injury)')
            plt.xticks(rotation = 45, fontsize=5)
            plt.yticks(rotation = 0, fontsize=5)
            sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='type_of_injury', palette='rocket')
            plt.legend(title='Type of Injury', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks([])
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)
    
    with tab3:
            plt.figure(figsize=(10, 8))
            plt.xlabel('conflict from 2000 to 2023')
            plt.ylabel('Age')
            plt.title('Age vs. Date of Death (citizenship)')
            plt.xticks(rotation = 45, fontsize=5)
            plt.yticks(rotation = 0, fontsize=5)
            sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age', hue='citizenship', palette='Set2')
            plt.legend(title='Citizenship', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks([])
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)
    
    with tab4:
            plt.figure(figsize=(10, 8))
            plt.xlabel('conflict from 2000 to 2023')
            plt.ylabel('Age')
            plt.title('Age vs. Date of Death (killed by)')
            plt.xticks(rotation = 45, fontsize=5)
            plt.yticks(rotation = 0, fontsize=5)
            sns.scatterplot(data = fatalities_df, x = 'date_of_death', y= 'age',  hue='killed_by', palette='husl')
            plt.legend(title='killed by:', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks([])
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)

    st.write('''
         From the following scatterplots we can draw several conclusions:
            - a preliminary analysis shows that most of the victims are men, and a large number of deaths occurred between the years 2002 and 2008.
            - the leading cause of death is gunfire, regardless of the period or years of the victim. the second most common cause of death is from stabbing, hit by veichle, and explosion. The latter especially were prevalent between 2002 and 2008.
            - from the third graph we can see that Palestinian casualties far outnumber Israeli casualties. This phenomenon, though, was not so obvious at the beginning of the conflict, where it would in fact appear to be the reverse
            - from the last graph instead, it shows how most of the casualties were caused by Israeli security forces, while Palestinian citizens killed many Israelis early in the war when they rebelled against the Israeli government's expansionist policy.
             ''')
########################################### insert code of encoding and add the graph

    st.subheader('Correlation by encoding categorical variables')
    st.write("When variables are categorical (represented by labels or categories instead of numerical values) encoding is imperative. Encoding of categorical variables is the process of converting these variables into a numerical form without losing the informational meaning of the original categories. This step is crucial to enable the application of correlation calculation algorithms based on mathematical operations.")
    import copy
    data = copy.deepcopy(fatalities_df)
    cat_cols = ['citizenship', 'event_location','event_location_district', 'event_location_region','gender', 
            'place_of_residence','place_of_residence_district', 'type_of_injury','killed_by']
    for x in cat_cols:
        print("column",x,"has nulls:",data[x].hasnans,",count:",data[x].isnull().sum())
        data[x+"_cat"] = pd.CategoricalIndex(data[x]).codes
    
    corr_matrix = data[['citizenship_cat', 'event_location_cat',
        'event_location_district_cat', 'event_location_region_cat',
        'gender_cat', 'place_of_residence_cat',
        'place_of_residence_district_cat', 'type_of_injury_cat',
        'killed_by_cat']].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set(style="whitegrid")
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, annot_kws={"size": 8}, ax=ax)
    plt.title("Correlation between categorical variables")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

elif current_tab == "Exploratory Data Analysis":
    st.header("Sezione")

elif current_tab == "Modelling":
    st.header("Sezione")



















########################################### test



