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
from streamlit_folium import folium_static

########################################### Import Dataset and set tabs

fatalities_df = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')
tab_names = ["Introduction", "Cleaning and Correlation", "Exploratory Data Analysis", "Modelling"]
current_tab = st.sidebar.selectbox("Summary", tab_names)

########################################### Main Titles and introduction
if current_tab == "Introduction":
    st.markdown("<h1 style='text-align: center;'>Exploring Terrorism Victim Data: A Look into the Israeli-Palestinian Conflict</h1>", unsafe_allow_html=True)
    st.subheader('Programming and Database Final Project')
    st.markdown('''
                **Author**: Gabriele Durante\n
                [GitHub](https://github.com/exdsgift) | [LinkedIn](https://www.linkedin.com/in/gabrieledurante/)
                ''')

    st.write('The dataset is available at the following [Kaggle link](https://www.kaggle.com/datasets/willianoliveiragibin/fatalities-in-the-israeli-palestinian). It reports data on victims of terrorism in Israel and the war in Palestine. This project aims to analyse events and fatalities by:')
    st.write('''
            - Year, month, and day of the month of events that led to fatalities
            - Victim profiles such as age, gender, citizenship, participation in hostilities etc
            - Event locations, location districts, and location regions
            - Type of injury, type of ammunition, and party responsible for killings, among others.
            ''')

    selected_columns = st.multiselect('Explore the dataset by selecting columns', fatalities_df.columns)
    if selected_columns:
        filtered_df = fatalities_df.loc[:, selected_columns]
        st.dataframe(filtered_df.head(51))
    else:
        st.dataframe(fatalities_df.head(51))
    
    
########################################### From plt to image function, design def function

    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf

    graph_image = plt_to_image(plt)

########################################### insert containers using with to add graphs
elif current_tab == "Cleaning and Correlation": 
    st.title("Cleaning and Correlation")
   
    st.subheader('Cleaning: NA values and substitutions') 
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')
    code_block = """
    {(fatalities_df['age'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['gender'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['took_part_in_the_hostilities'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['place_of_residence'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['place_of_residence_district'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['type_of_injury'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['ammunition'].isna().sum()/len(fatalities_df))*100:.2f}%")
    {(fatalities_df['notes'].isna().sum()/len(fatalities_df))*100:.2f}%")
    """
    st.code(code_block, language='python')
    
    st.markdown('''
                based on this information, the data will be rearranged as follows:
- **Replace** missing values for the variable **age** with the **mean value** (1.16%).
- Missing values for the variables **sex**, **place_of_residence**, **place_of_residence_district**, **type_of_location**, and **notes** are **replaced** with the **mode value**. this is so as not to change the distribution of the data too much.
- **Remove** the variables **has_participated_in_hostilities**, **ammunition** from the variable itself. The percentage of null values is excessive and would not lead to useful information for the entire population of the dataset.
                ''')
    code_block2 = """
    age_mean = fatalities_df['age'].mean()
    fatalities_df['age'].fillna(age_mean, inplace=True)
    
    gender_mode = fatalities_df['gender'].mode()[0]
    fatalities_df['gender'].fillna(gender_mode, inplace=True)
    place_of_residence_mode = fatalities_df['place_of_residence'].mode()[0]
    fatalities_df['place_of_residence'].fillna(place_of_residence_mode, inplace=True)
    place_of_residence_district_mode = fatalities_df['place_of_residence_district'].mode()[0]
    fatalities_df['place_of_residence_district'].fillna(place_of_residence_district_mode, inplace=True)
    type_of_injury_mode = fatalities_df['type_of_injury'].mode()[0]
    fatalities_df['type_of_injury'].fillna(type_of_injury_mode, inplace=True)
    notes_mode = fatalities_df['notes'].mode()[0]
    fatalities_df['notes'].fillna(notes_mode, inplace=True)
    
    fatalities_df.drop(['took_part_in_the_hostilities', 'ammunition'], axis=1, inplace=True)
    """
    st.code(code_block2, language='python') 
    
    
    
    
    
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
            plt.xlabel('Victims from 2000 to 2023')
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
            plt.xlabel('Victims from 2000 to 2023')
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
            plt.xlabel('Victims from 2000 to 2023')
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
            plt.xlabel('Victims from 2000 to 2023')
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
    st.write('This analysis shows that there is a positive correlation between the location of the event and the country of residence, instead a particularly negative one between the citizenship of the victims and the instigator of the killing.')
    
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
    set_color = sns.color_palette("RdBu", as_cmap=True)
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, annot_kws={"size": 8}, cmap=set_color, ax=ax)
    plt.title("Correlation between categorical variables")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    









##################################### new tab, adding fun
elif current_tab == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf
    
##################################### Nationalities graphs

    st.subheader('Nationality graphs')

    fatalities_df_2 = fatalities_df[~fatalities_df['citizenship'].isin(['Jordanian', 'American'])]
    fatalities_df_2['citizenship'].value_counts()

    plt.figure(figsize=(10,6))
    sns.set(style = "white")
    plt.title('Deaths by nationality (Palestinian and Israeli)')
    plt.xlabel("citizenship")
    plt.ylabel("fatalities")
    plt.xticks(rotation = 0, fontsize=10)
    plt.yticks(rotation = 45, fontsize=10)

    custom_palette = sns.color_palette("viridis", n_colors=2)
    plot1 = sns.countplot(x='citizenship', data = fatalities_df_2, palette = custom_palette, hue='citizenship', legend=False)

    for p in plot1.patches:
        plot1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)
    
    
    fatalities_df['date_of_death'] = pd.to_datetime(fatalities_df['date_of_death'])
    fatalities_df['year'] = fatalities_df['date_of_death'].dt.year
    fatality_by_year = fatalities_df.groupby('year').size().reset_index(name='fatalities')
    plt.figure(figsize=(10, 6))
    plt.title('Death trends from 2000 to 2023: Total, Palestinian and Israeli')
    plt.xlabel('years')
    plt.ylabel('deaths')
    plt.xticks(fatality_by_year['year'], rotation = 45, fontsize=10)
    plt.yticks(rotation = 0, fontsize=10)
    viridis_palette = sns.color_palette('viridis', n_colors=3)
    selected_colors = [viridis_palette[0], viridis_palette[1], viridis_palette[2]]
    fatality_by_year = fatalities_df.groupby('year').size().reset_index(name='fatalities')
    plot4 = sns.lineplot(x='year',
                        y='fatalities',
                        data=fatality_by_year,
                        markers=True,
                        color='green',
                        linestyle='dashed',
                        linewidth=1,
                        label='Fatalities Trend')

    palestinian_data = fatalities_df[fatalities_df['citizenship'] == 'Palestinian']
    palestinian_fatalities_by_year = palestinian_data.groupby('year').size().reset_index(name='palestinian_fatalities')
    sns.lineplot(x='year',
                y='palestinian_fatalities',
                data = palestinian_fatalities_by_year,
                markers=True,
                color='purple',
                label='Palestinian Fatalities Trend')
    for i, txt in enumerate(palestinian_fatalities_by_year['palestinian_fatalities']):
        plt.annotate(txt, (palestinian_fatalities_by_year['year'][i], txt), textcoords="offset points", xytext=(0, 5), ha='center', color='black', fontsize = 9)


    israeli_data = fatalities_df[fatalities_df['citizenship'] == 'Israeli']
    israeli_fatalities_by_year = israeli_data.groupby('year').size().reset_index(name='israeli_fatalities')
    sns.lineplot(x='year',
                y='israeli_fatalities',
                data=israeli_fatalities_by_year,
                markers=True,
                color='blue',
                label='Israeli Fatalities Trend')
    for i, txt in enumerate(israeli_fatalities_by_year['israeli_fatalities']):
        plt.annotate(txt, (israeli_fatalities_by_year['year'][i], txt), textcoords="offset points", xytext=(0, 5), ha='center', color='black', fontsize = 9)

    plt.legend(title='Death Trend:', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)

##################################### Gender graphs

    st.subheader('Gender graphs')


##################################### WordCloud graphs
    st.subheader('WordCloud graph')
    
    words_in_note = ''.join(fatalities_df['notes'].astype(str))
    wordcloud = WordCloud(width=1200, height=800,
                min_font_size = 10, max_font_size=150).generate(words_in_note)
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Most frequent words in the dataset notes')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)



##################################### Location graphs

    st.subheader('Location graphs')
    
    top_locations = fatalities_df['event_location'].value_counts().index[:20]
    filtered_df = fatalities_df[fatalities_df['event_location'].isin(top_locations)]
    location_counts = filtered_df['event_location'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.title('Regions with the most recorded deaths (2000-2023)')
    plt.ylabel('Region')
    plt.xlabel('Deaths')
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    custom_palette = sns.color_palette("inferno", n_colors=20)
    sns.barplot(x=location_counts.values,
                y=location_counts.index,
                palette=custom_palette,
                hue=location_counts.index,
                legend=False)

    for i, v in enumerate(location_counts.values):
        plt.text(v + 10, i, str(v), color='black', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.grid(True, axis='x')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)
    
























elif current_tab == "Modelling":
    st.header("Modelling")



















########################################### test


