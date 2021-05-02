#!/usr/bin/env python
# coding: utf-8

# # Implementing a course recommendation system
# Use-Cases:
# - To any online edu course platform like Coursera, Udemy etc (They all are having recommendation system on their platform)
# 
# ## Different way to build a recommendation engine:
# - ### Association Rule (mainly used for e-commerce using apriori algorithm with support, lift, and confidence). ex: If you purchase A, you might purchase B and C, etc.
# 
# - ### Collaborative Filtering
#  - Mainly of two types: 1. User-based similarity and 2. Item-based similarity
#  - User-based similarity are good for analysis of past data. Once a new user comes, it will not make any sense as this technique finds the similarity between the two users or multiple users based on "how they have rated a particular movies".
#  - Item-based similarity is based on items not on users similarity.
#  
# - ### Matrix Factorization
# 
# - ### Hybrid Recommendation technique
# 
# 

# ## Data to use: Udemy Course Dataset from Kaggle

# ## We will be seeing:
# - Cosine similarity 
# - Linear similarity (much faster)
# 
# ### Workflow
# - Load the data
# - Vectorize the dataset
# - Finding the cosine similarity matrix
# - ID and score
# - Recommendation to the user

# In[101]:


import pandas as pd
import neattext.functions as nfx # to clean the text-based records


# In[102]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


# In[103]:


data = pd.read_csv("udemy_courses.csv")


# In[104]:


data


# ## focus on the "course_title" column.
# our center point. 
# 
# - Needs to remove the special char, stopwords, etc from the records in that column.

# In[105]:


data['course_title']


# In[106]:


data['cleaned_course_title'] = data['course_title'].apply(nfx.remove_stopwords)


# In[107]:


data['cleaned_course_title']


# In[108]:


data['cleaned_course_title'] = data['cleaned_course_title'].apply(nfx.remove_special_characters)


# In[109]:


data['cleaned_course_title']


# In[110]:


data[['course_title','cleaned_course_title']]


# ## Let's build the count vectorizer matrix

# In[111]:


count_vectorization = CountVectorizer()
count_vectorizer_matrix = count_vectorization.fit_transform(data['cleaned_course_title'])


# In[112]:


count_vectorizer_matrix


# In[113]:


# sparse to dense
count_vectorizer_matrix.todense()


# ## Let's find the cosine similarity now

# In[114]:


cosine_similarity_matrix = cosine_similarity(count_vectorizer_matrix)


# In[115]:


cosine_similarity_matrix


# In[116]:


cosine_similarity_matrix[0:10]


# In[117]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


sns.heatmap(cosine_similarity_matrix[0:10], annot = True)


# In[119]:


plt.figure(figsize = (10,12))
sns.heatmap(cosine_similarity_matrix[0:10], annot = True)
plt.show()


# ## Getting the course index

# In[120]:


course_index = pd.Series(data.index, index = data['course_title']).drop_duplicates()


# In[121]:


course_index


# In[122]:


course_index['Learn and Build using Polymer']


# In[123]:


idx = course_index['Financial Modeling for Business Analysts and Consultants']


# In[124]:


scores = list(enumerate(cosine_similarity_matrix[idx]))


# In[125]:


scores


# ## get the highest score

# In[126]:


sort_scores = sorted(scores, key = lambda x:x[1], reverse = True)


# In[127]:


sort_scores


# In[128]:


#omit number 1 and show top 5 scores
sort_scores[1:6]


# ## fetch the course index and name and other details

# In[129]:


selected_course_indices = [i[0] for i in sort_scores[1:6]]


# In[130]:


selected_course_indices


# In[131]:


data['course_title'].iloc[selected_course_indices]


# In[132]:


recommended_results = pd.DataFrame(data['course_title'].iloc[selected_course_indices])


# In[133]:


recommended_results


# In[134]:


selected_course_scores = [i[1] for i in sort_scores[1:6]]


# In[135]:


recommended_results['similarity_score'] = selected_course_scores


# In[136]:


recommended_results


# ## Abstract

# In[149]:


def recommendation_results(title, num_of_recommendations = 5):
    #ID
    idx = course_index[title]
    
    # search cosine similiarty matrix
    scores = list(enumerate(cosine_similarity_matrix[idx]))
    
    # sort the score
    sort_scores = sorted(scores, key = lambda x:x[1], reverse = True)
    
    # recommendation
    selected_course_indices = [i[0] for i in sort_scores[1:]]
    selected_course_scores = [i[1] for i in sort_scores[1:]]
    
    # result
    result = data['course_title'].iloc[selected_course_indices]
    recommended_results = pd.DataFrame(result)
    recommended_results['similarity_score'] = selected_course_scores
    
    return recommended_results.head(num_of_recommendations)


# In[154]:


recommendation_results('Financial Modeling 101')


# In[ ]:




