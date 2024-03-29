
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
from streamlit_folium import st_folium
from streamlit_folium import folium_static

########################################### Import Dataset and set tabs for chapters

fatalities_df = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')
tab_names = ["Introduction", "Cleaning and Correlation", "Exploratory Data Analysis", "Modeling with ML algorithms"]
current_tab = st.sidebar.selectbox("Summary", tab_names)
st.sidebar.markdown(
    """
    **Gabriele Durante**  
    [GitHub](https://github.com/exdsgift)  [LinkedIn](https://www.linkedin.com/in/gabrieledurante/)  [gabriele.durante@studenti.univr.it](mailto:gabriele.durante@studenti.univr.it)
    """
)
####################################################################################### INTRODUCTION

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

####################################################################################### CLEANING AND CORRELATION

elif current_tab == "Cleaning and Correlation": 
    st.title("Cleaning and Correlation")
   
    st.subheader('Cleaning: NA values and substitutions')
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')
    
    tab1, tab2, tab3, tab4 = st.tabs(["NA's values", "Cleaning", "-", "-"]) 
    
    with tab1:
    
        columns_to_check = ['age', 'gender', 'took_part_in_the_hostilities', 'place_of_residence', 'place_of_residence_district', 'type_of_injury', 'ammunition', 'notes']
        def calculate_na_percentage(column):
            return (fatalities_df[column].isna().sum() / len(fatalities_df)) * 100
        for column in columns_to_check:
            percentage_na = calculate_na_percentage(column)
            st.code(f"Percentage of {column} NA value: {percentage_na:.2f}%")

    with tab2:
        st.markdown('''
                based on this information, the data will be rearranged as follows:
                - **Replace** missing values for the variable **age** with the **mean value** (1.16%).
                - Missing values for the variables **sex**, **place_of_residence**, **place_of_residence_district**, **type_of_location**, and **notes** are **replaced** with the **mode value**. this is so as not to change the distribution of the data too much.
                - **Remove** the variables **has_participated_in_hostilities**, **ammunition** from the variable itself. The percentage of null values is excessive and would not lead to useful information for the entire population of the dataset.
                                ''')    
############################################ add graphs from the EDA side (same code)
    
    st.divider() 
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
            plt.legend(title= 'Gender')
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
            plt.legend(title='Type of Injury',loc='upper right')
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
            plt.legend(title='Citizenship')
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
            plt.legend(title='killed by:')
            plt.xticks([])
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)

    st.caption('''
         From the following scatterplots we can draw several conclusions:
            - a preliminary analysis shows that most of the victims are men, and a large number of deaths occurred between the years 2002 and 2008.
            - the leading cause of death is gunfire, regardless of the period or years of the victim. the second most common cause of death is from stabbing, hit by veichle, and explosion. The latter especially were prevalent between 2002 and 2008.
            - from the third graph we can see that Palestinian casualties far outnumber Israeli casualties. This phenomenon, though, was not so obvious at the beginning of the conflict, where it would in fact appear to be the reverse
            - from the last graph instead, it shows how most of the casualties were caused by Israeli security forces, while Palestinian citizens killed many Israelis early in the war when they rebelled against the Israeli government's expansionist policy.
             ''')
    
########################################### insert code of encoding and add the graph
    st.divider()
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

####################################################################################### EDA, introduction of this part and adding the global trend graph
elif current_tab == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf
    st.write('The first important piece of data investigated through a historical analysis of deaths is the trend regarding the deaths of Palestinians and Israelis, combined within a graph. One can already sense how there is a great disparity in these values and how there are major peaks during 2014, a year of high tension between the two countries. in fact, in this year, the conflict in Gaza between July and August caused the deaths of thousands of Palestinians and displaced about half a million people. Palestinian attacks on Israeli civilians and security forces increased, as did incidents of settler violence that resulted in Palestinian casualties and injuries.')
    st.markdown('''
                [TheGuardian](https://www.theguardian.com/world/2015/mar/27/israel-kills-more-palestinians-2014-than-any-other-year-since-1967) - [TheGuardian](https://www.theguardian.com/world/2024/jan/08/the-numbers-that-reveal-the-extent-of-the-destruction-in-gaza) - [United Nations](https://news.un.org/en/story/2015/06/502282)
                ''')
    
    fatalities_df['date_of_death'] = pd.to_datetime(fatalities_df['date_of_death'])
    fatalities_df['year'] = fatalities_df['date_of_death'].dt.year
    fatality_by_year = fatalities_df.groupby('year').size().reset_index(name='fatalities')
    plt.figure(figsize=(10, 6))
    plt.title('Death trends from 2000 to 2023: Total, Palestinian and Israeli')
    plt.xlabel('years')
    plt.ylabel('deaths')
    plt.xticks(fatality_by_year['year'], rotation = 45, fontsize=10)
    plt.yticks(rotation = 45, fontsize=10)
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

    plt.legend(title='Death Trend:')
    plt.grid(True, axis='y')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)

##################################### Nationalities graphs

    st.divider()
    st.subheader('Data on the nationality of the victims and murderers')
    st.write('The first variables to be examined, were those inherent in the nationality of the victims and perpetrators. This is used to get an initial idea of the forces deployed by both countries.')
    
##################################### definig tabs for this chapter

    tab1, tab2, tab3, tab4 = st.tabs(["Victims", "Perpetrators", "Weapons used", "Casualities by entity and type of injury"])

##################################### adding graphs for these tabs    
    with tab1:
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
            st.caption('From this chart we observe that most of the victims are Palestinians, while the Israelis who died are a significant minority (about one-tenth of those Palestinians)')
    
    with tab2:
            killed_count = fatalities_df['killed_by'].value_counts()
            plt.figure(figsize=(10, 6))
            custom_palette = sns.color_palette("viridis", n_colors=3)
            sns.set(style = "white")
            plt.title('Victims for perpetrators of killings')
            plt.xlabel('Responsible for the killings')
            plt.ylabel('Total casualties')
            plt.xticks(rotation = 0, fontsize=10)
            plt.yticks(rotation = 45, fontsize=10)
            bars = plt.bar(killed_count.index, killed_count.values)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)
            st.caption('Instead, studying the distribution of persecutors, we note how Israel carries out all killings through the army, while the among Palestinians there are only killings carried out by civilians. This is a great point of reflection on the difference in counteroffensive power.')

    with tab3:
            fatalities_df_original = pd.read_csv('/Users/gabrieledurante/Documents/uni/data science UNIVR - Master Degree/programming/datasets _for _final_project/fatalities_isr_pse_conflict_2000_to_2023.csv')
            ammunition_kll = fatalities_df_original['ammunition'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.set(style="white", palette="Set2")
            ax1 = ammunition_kll.head(10).plot(kind='bar')
            plt.title('Most used weapons in conflict')
            plt.xlabel('')
            plt.ylabel('Deaths')
            plt.xticks(rotation=90, fontsize=10)
            plt.yticks(rotation=45, fontsize=10)
            for p in ax1.patches:
                ax1.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)
            graph_image = plt_to_image(plt)
            st.image(graph_image)
            st.caption('From what emerges from an analysis of the equipment used, we note how they can be traced back to urban and civil conflict. Note how the explosive belt killings are high, attributable to attacks on convoys usually.')                
            
    with tab4:
            fatalities_df['casualty_count'] = 1
            killed_injury = fatalities_df.groupby(['killed_by', 'type_of_injury'])['casualty_count'].sum().reset_index()
            killed_injury_pivot = killed_injury.pivot(index='killed_by', columns='type_of_injury', values='casualty_count').fillna(0)
            plt.figure(figsize=(10, 6))
            sns.set(style = "white",  palette = sns.color_palette("Set2"))
            ax = killed_injury_pivot.plot(kind='bar', stacked=True)
            plt.title('Distribution of casualties by Entity and type of injury')
            plt.xlabel('Entity Responsible for Killings')
            plt.ylabel('Total Casualties')
            plt.xticks(rotation = 0, fontsize=10)
            plt.yticks(rotation = 45, fontsize=10)
            plt.legend(title='Type of Injury', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
            graph_image = plt_to_image(plt)
            st.image(graph_image)
            st.caption('From this chart it is clear that the Israelis mainly use modern weapons of war such as rockets and firearms, while the Palestinians still use more radical methods such as explosive belts and edged weapons to counterattack.')        

##################################### Gender graphs
    st.divider()
    st.subheader('Data on biological sex of victims')
    st.write('In a war, in addition to questioning the political and religious causes of the conflict, it is also important to question the protagonists of the conflict. In the following graphs we analyze the age of the victims and their biological sex, as well as go on to study the distribution function.')

    tab1, tab2, tab3 = st.tabs(["Violin", "Gender differences", "Distribution"])
    
    with tab1:
        plt.figure(figsize=(10, 6))
        plt.title('Fatalities by Age vs. Gender')
        plt.xlabel('ages')
        plt.ylabel(None)
        plt.xticks(rotation = 0, fontsize=10)
        sns.set(style="white")
        custom_palette = sns.color_palette("viridis", n_colors=2)
        sns.violinplot(data=fatalities_df, x="age", y='gender', hue = "gender", palette = custom_palette, legend=True)
        plt.legend(title='gender')
        plt.grid(True, axis='x')
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)

    with tab2:
        plt.figure(figsize=(10, 6))
        sns.set(style="white", palette="Set1")
        plt.title("Fatalities by age and gender during the conflict")
        plt.xlabel("Age")
        plt.ylabel("Deaths")
        plt.legend(['Woman', 'Man'])
        plt.xticks(rotation = 0, fontsize=10)
        plt.yticks(rotation = 0, fontsize=10)
        ax = sns.histplot(data=fatalities_df, x='age', hue='gender', bins=28)
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)
    
    with tab3:
        plt.figure(figsize=(10, 6))
        sns.set(style="white", palette="Set2")
        sns.histplot(fatalities_df['age'], kde=True)
        plt.title('Age Distribution of victims during the conflict')
        plt.xlabel('Age')
        plt.ylabel("Deaths")
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)
    st.caption('From this information, it can be guessed that men are more exposed in this conflict than women, although the latter have a more even distribution of deaths by age group.')

##################################### WordCloud graphs

    st.divider()
    st.subheader("WordCloud graph on dataset's note")
    
    words_in_note = ''.join(fatalities_df['notes'].astype(str))
    wordcloud = WordCloud(width=1200, height=800,
                min_font_size = 10, max_font_size=150).generate(words_in_note)
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Most frequent words in the dataset notes')
    plt.axis('off')
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)
    st.write('This graph is a WordCloud, which is a visual way to represent the most frequent words in a given dataset. In this case, the WordCloud was created based on the variable "Notes." This variable contains information about the dynamics of conflict-related deaths, and the largest words in the are those that appear most frequently in the notes. This can help to quickly identify the most common themes or terms associated with the dynamics of deaths in the conflict.')

##################################### Location graphs

    st.divider()
    st.subheader('Location data and Folium Map')
    st.write('By having data on the location where the deaths occurred, it was studied which areas were affected by urban guerrilla warfare. Then after determining which were the most affected by the conflict, using the coordinates and the folium package, a chart was created containing the map with the hazard areas according to the deaths.')
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

    ##################################### Folium chart and code
    st.write('Reporated on the map are the 10 cities most affected by the conflict derived from the graph above.')
    
    tab1, tab2 = st.tabs(["Folium chart", "Script"])
    
    with tab2:
        st.code('''
            from streamlit_folium import st_folium
            from streamlit_folium import folium_static
            
            # color selection
            
            def get_color(fatalities):
                if fatalities > 500:
                    return 'red'
                elif fatalities > 250:
                    return 'orange'
                elif fatalities > 100:
                    return 'yellow'
                else:
                    return 'green'
            
            # build up the graph
            
            base_map = folium.Map(location=[32, 34.75], zoom_start=8)
            for district, coords in district_coords.items():
                fatalities = district_fatalities.get(district, 0)
                folium.Marker(location = coords,
                tooltip = f'District: {district},
                Deaths: {fatalities}',
                icon = None).add_to(base_map)
                folium.Circle(location=coords,
                radius=np.sqrt(fatalities) * 1200,
                color=get_color(fatalities),
                fill=True,
                fill_color=get_color(fatalities),
                fill_opacity=0.6,).add_to(base_map)
            folium.LayerControl().add_to(base_map)
            st.data = st_folium(base_map, width=800, height=480)
            ''')
    
    with tab1:
        district_coords = {
        'Gaza': [31.5, 34.466667],
        'Hebron': [31.532569, 35.095388],
        'Jenin': [32.457336, 35.286865],
        'Nablus': [32.221481, 35.254417],
        'Ramallah': [31.902922, 35.206209],
        'Bethlehem': [31.705791, 35.200657],
        'Tulkarm': [32.308628, 35.028537],
        'Jericho': [31.857163, 35.444362],
        'Rafah': [31.296866, 34.245536],
        'Khan Yunis': [31.346201, 34.306286]
        }
        district_fatalities = fatalities_df.groupby('event_location_district').size()
        def get_color(fatalities):
            if fatalities > 500:
                return 'red'
            elif fatalities > 250:
                return 'orange'
            elif fatalities > 100:
                return 'yellow'
            else:
                return 'green'
        base_map = folium.Map(location=[32, 34.75], zoom_start=8)
        for district, coords in district_coords.items():
            fatalities = district_fatalities.get(district, 0)
            folium.Marker(location = coords, tooltip = f'District: {district}, Deaths: {fatalities}', icon = None).add_to(base_map)
            folium.Circle(location=coords, radius=np.sqrt(fatalities) * 1200, color=get_color(fatalities), fill=True, fill_color=get_color(fatalities), fill_opacity=0.6,).add_to(base_map)
        folium.LayerControl().add_to(base_map)
        st.data = st_folium(base_map, width=800, height=480)
        
####################################################################################### MODELING

elif current_tab == "Modeling with ML algorithms":
    st.header("Modeling using Machine Learning algorithms")
    
########################################### PCA Analysis and uploading datasets variations

    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    st.write('Having almost exclusively categorical variables in the dataset, the data relating to the encoding of the categorical variables were used in the modeling phase. They were first processed through PCA Analysis to reduce the size of the data, and then processed through the KMeans clustering algorithm to identify groups based on similarities.')
    
    def plt_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return buf
    st.subheader('PCA Analysis using Categorical Values')
    st.write('Principal Component Analysis (PCA) is a dimensionality reduction technique that is often used to simplify data while retaining the most meaningful information.')
    
    fatalities_df = pd.read_csv('fatalities_isr_pse_conflict_2000_to_2023.csv')
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
    
    ######
    
    import copy
    data = copy.deepcopy(fatalities_df)
    cat_cols = ['citizenship', 'event_location','event_location_district', 'event_location_region','gender', 
            'place_of_residence','place_of_residence_district', 'type_of_injury','killed_by']
    for x in cat_cols:
        print("column",x,"has nulls:",data[x].hasnans,",count:",data[x].isnull().sum())
        data[x+"_cat"] = pd.CategoricalIndex(data[x]).codes
    mydata= data[['citizenship_cat', 'event_location_cat',
        'event_location_district_cat', 'event_location_region_cat',
        'gender_cat', 'place_of_residence_cat',
        'place_of_residence_district_cat', 'type_of_injury_cat',
        'killed_by_cat']]
    components_range=range(1,len(mydata.columns)+1)
    for n in components_range:
        pca = PCA(n_components=n)
        pca.fit(mydata)
        print(n,"components, variance ratio=",pca.explained_variance_ratio_)
        
    ######
    
    pca = PCA(n_components=len(mydata.columns))
    pca.fit(mydata)
    explained_variance=pca.explained_variance_ratio_
    cumulative_explained_variance=np.cumsum(pca.explained_variance_ratio_)
    plt.plot(components_range, explained_variance,marker='o', label='Explained Variance per Component')
    plt.plot(components_range, cumulative_explained_variance,marker='+', label='Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Elbow Diagram for fatalities PCA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    graph_image = plt_to_image(plt)
    st.image(graph_image)

    st.write('The elbow graph shows how much of the total variance is explained by the first N principal components. In this case, the N number of principal components is 2.')
    
########################################### KMeans clusters analysis

    st.subheader('K-means Clustering')
    st.write('To evaluate the actual effectiveness of the clustering algorithm, we examine the silhouette coefficient to evaluate the cohesion [-1, 1].')
  
    tab1, tab2 = st.tabs(["PCA2", "PCA9"])
    
    with tab1:
        pca = PCA(n_components=2)
        pca.fit(mydata)
        pca_data=pca.fit_transform(mydata)
    
        kmeans_2 = KMeans(n_clusters=2, random_state=20)
        kmeans_2.fit(mydata)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_2.labels_, cmap='Accent')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 2')
        plt.legend().set_visible(False)
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)

        from sklearn.metrics import silhouette_score
        silhouette_result = silhouette_score(mydata, kmeans_2.labels_)
        st.write("Silhouette coefficient for 2 clusters on data:", silhouette_result)
        st.write('Using N = 2 as the number of clusters as suggested by the analysis done earlier, we note how the coefficient si silhouette acquires a high average value.')


    with tab2:
        pca = PCA(n_components=9)
        pca.fit(mydata)
        pca_data9 = pca.fit_transform(mydata)
        
        kmeans_9 = KMeans(n_clusters=9, random_state=20, n_init=10)
        kmeans_9.fit(mydata)
        plt.scatter(pca_data9[:, 0], pca_data9[:, 1], c=kmeans_9.labels_, cmap='Set2')
        plt.xlabel('Main component PCA1')
        plt.ylabel('Main component PCA2')
        plt.title('K-means Clustering (n = 9)')
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)
        silhouette_result = silhouette_score(mydata, kmeans_9.labels_)
        st.write("Silhouette coefficient for 9 clusters on data:", silhouette_result)
        st.write('Test to try the maximum number (9) of pca components and then setting 9 as the number of clusters.')
    
    st.divider()
    st.subheader('Finding the best silhouette coefficient using loops.')
    st.write('By using a loop, numerous attempts can be made in order to find the number of clusters that maximizes the coefficient. Only a few tests are given below (maximum value N = 1000), as the computational power using too high values of N is too much.')

    tab1, tab2, tab3 = st.tabs(["PCA100", "PCA1000", 'Loop script'])

    with tab1:
        kmeans_100 = KMeans(n_clusters=100, random_state=20)
        kmeans_100.fit(mydata)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_100.labels_, cmap='Set2')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 100')
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)
        silhouette_result = silhouette_score(mydata, kmeans_100.labels_)
        st.write("Silhouette coefficient for 100 clusters on data:", silhouette_result)
    
    with tab2:
        kmeans_1000 = KMeans(n_clusters=1000, random_state=20)
        kmeans_1000.fit(mydata)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_1000.labels_, cmap='Set2')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 1000')
        plt.show()
        graph_image = plt_to_image(plt)
        st.image(graph_image)
        silhouette_result = silhouette_score(mydata, kmeans_1000.labels_)
        st.write("Silhouette coefficient for 1000 clusters on data:", silhouette_result)
    
    with tab3:
        st.code('''
                # Set the maximum number of clusters you wish to explore.
                max_clusters = 1000

                # Initialize variables to keep track of the best results:
                best_num_clusters = 2  # Start with a reasonable value
                best_silhouette_score = -1  # Initialize with an impossible value

                # find the value using loop
                for num_clusters in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=num_clusters, random_state=20) # the random state number is arbitrarial but for different tests it has to be the same (centroids locations)
                    kmeans_labels = kmeans.fit_predict(mydata)
                    silhouette_avg = silhouette_score(mydata, kmeans_labels)
                    
                    # Update if the loop find a better solution
                    if silhouette_avg > best_silhouette_score:
                        best_silhouette_score = silhouette_avg
                        best_num_clusters = num_clusters

                print(f"The maximum number of clusters with the best silhouette coefficient is: {best_num_clusters}")
                print(f"Associated silhouette coeficient: {best_silhouette_score}")
                
                ### with max_clusters = 1000, the coefficient is equal to 0.7939302027453841 and the max numnber is 1000 
                ### with max_cluserts = 10, the coefficent is equal to 0.5831316118613485 and the max number is 2
                ### with max_clusters = 100, the coefficient is equal to 0.7148252056599078 and the max numnber is 100
                ''')
    st.write('''
            By analyzing the results obtained from the K-means model using a loop cycle, we identified an optimal number of clusters of 1000, accompanied by a significant silhouette coefficient of 0.71.

            The high silhouette coefficient suggests a valid separation between clusters and internal consistency. The numerosity of the clusters, however, raises questions about the true complexity of the data.
             ''')