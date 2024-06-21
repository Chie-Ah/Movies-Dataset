#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import re
import seaborn as sns
df = pd.read_csv("movies.csv")
df.head(10)


# In[60]:


df.shape


# In[64]:


df.duplicated().sum()


# In[65]:


df = df.drop_duplicates()


# In[61]:


#checking percentage of missing values on each column
df.isnull().sum()/df.shape[0]*100


# In[63]:


#to drop Gross column since it has a striking percentage of missing values 
df = df.drop(columns = 'Gross')


# In[ ]:


WORKING WITH THE RunTime Column


# In[66]:


df['RunTime']


# In[67]:


#getting mean to fill irregular missing values on this column
df['RunTime'].mean()


# In[68]:


# mean value(mv) approx to 1decimal place
mv = 69.0
df['RunTime'].fillna(mv, inplace=True)
df['RunTime']


# In[ ]:


UNTO CLEANING THE VOTES COLUMN


# In[69]:


df['VOTES']


# In[70]:



# 1. Removing commas from the figures
df.loc[:, 'VOTES'] = df['VOTES'].str.replace(',', '')

# 2. Converting values to numeric to enable replacement of null values
df.loc[:, 'VOTES'] = pd.to_numeric(df['VOTES'], errors='coerce')

df['VOTES']


# In[71]:


#filling missing values through linear interpolation
df['VOTES'].interpolate(method='linear', inplace = True)
df['VOTES']


# In[72]:


df


# In[ ]:


WORKING WITH THE STARS COLUMN


# In[73]:


df['STARS'].info()


# In[74]:


#checking out unique values of this column, you'd notice we have directors and stars mixed
df['STARS'].nunique


# In[75]:


#creating seperate columns for Directors and STARS
def extract_director(text):
  if text is not None and 'Director:' in text:
    return text.split('|')[0].replace('Director:', '').strip()
  return None

def extract_stars(text):
  if text is not None and 'Stars:' in text:
    return text.split('|')[-1].replace('Stars:', '').strip()
  return None

# Apply the functions
df['Director'] = df['STARS'].apply(extract_director)
df['STARS'] = df['STARS'].apply(extract_stars)
df


# In[76]:


#taking out extra space from stars column
df.loc[:, 'STARS'] = df['STARS'].str.replace('\n', '')
df


# In[ ]:


NEXT, ONE-LINE COLUMN


# In[77]:


#almost perfect column, just needed to do away with the extra text, <\n>
df.loc[:, 'ONE-LINE'] = df['ONE-LINE'].str.replace('\n', '')
df


# In[ ]:


NEXT, RATING COLUMN


# In[79]:


#Went straight ahead to fill missing values
df['RATING'].interpolate(method='linear', inplace = True)
df['RATING']


# In[ ]:


NEXT, GENRE COLUMN


# In[80]:


#taking out extra text
df.loc[:, 'GENRE'] = df['GENRE'].str.replace('\n', '')
df


# In[ ]:


UNTO THE YEAR COLUMN


# In[81]:


#To better explore what data is on this column. I noticed a need to seperate the start and end year to help me do my visualizations freely at the end.
df['YEAR'].unique()


# In[83]:


#creating functions to seperate the dates into release year and, year movie release stopped.
def get_years(yr_str):
    if pd.isna(yr_str):
        return None, None  
    
    # Removing non-year characters 
    new_str = re.sub(r"^[^(]*", "", yr_str)
    
    # Extract years from cleaned string
    years = re.findall(r"\d{4}", new_str)

    if len(years) == 0:
        return None, None
    elif len(years) == 1:
        return years[0], None if '–' in new_str else years[0]
    else:
        return years[0], years[-1]


df['Release_Year'], df['End_Year'] = zip(*df['YEAR'].apply(get_years))
df


# In[86]:


# YEAR column no more of any use
df = df.drop('YEAR', axis=1)


# In[87]:


df


# In[ ]:


VISUALIZATIONS: EXPLORING MOVIE POPULARITY WITH A SCATTER PLOT


# In[112]:


#Plotting VOTES (y-axis) against RATING (x-axis) to see if there's a correlation between ratings and popularity (votes).
# Tip: Tighter clustering of points around a diagonal trend line indicates a stronger correlation. 

import matplotlib.pyplot as plt
import numpy as py

#columns for the plot
ratings = df['RATING']
votes = df['VOTES']

# Creating the scatter plot
plt.figure(figsize=(10, 6)) 
plt.scatter(ratings, votes, alpha=0.7)  

# labels and title
plt.xlabel('Rating')
plt.ylabel('Number of Votes')
plt.title('Movie Rating vs. Popularity (Votes)')

m, b = py.polyfit(ratings, votes, 1) 
plt.plot(ratings, m * ratings + b, color='red')  #to Plot the trendline

# Show plot
plt.grid(True)
plt.show()


# In[ ]:


Plot of movies released each year (Release_Year) to see how movie production has changed over time.


# In[114]:



# Count movies by release year
movies_per_year = df['Release_Year'].value_counts()

# Creating line chart
plt.figure(figsize=(16, 6))  
plt.plot(movies_per_year.index, movies_per_year.values)

# labels and title
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.title('Movie Releases Over Time')

#plot 
plt.xticks(rotation=90, ha="left")  
plt.grid(True)  # for gridlines
plt.show()


# In[ ]:


PLOT OF Average Rating OF Each Director, IN ORDER TO Get TOP Five Directors


# In[117]:


# average rating per director
director_ratings = (
    df.groupby('Director')['RATING']
    .mean() 
    .to_frame(name='Avg_Rating')
    .reset_index()
)

# Sort directors by average rating (descending) and select top 5
top_directors = director_ratings.sort_values(by='Avg_Rating', ascending=False).head(5)

# Create bar chart
plt.figure(figsize=(10, 6)) 
plt.bar(top_directors['Director'], top_directors['Avg_Rating'])

# labels and title
plt.xlabel('Director')  
plt.ylabel('Average Rating')
plt.title('Average Rating Distribution (Top 5 Directors)')


plt.xticks(rotation=45, ha='right')

# plot 
plt.grid(axis='y')  # Add gridlines on y-axis
plt.tight_layout()  # Adjusting layout to prevent overlapping elements
plt.show()


# In[ ]:




