import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load('../capstone2/models/doc2vec/enwiki_dbow/doc2vec.bin')  



def clean(data):
    data['description'] = [BeautifulSoup(text,'html.parser').get_text()\
        .replace('\n',' ').replace('\xa0','').replace("\'re"," are")\
        .replace("\'s"," is") for text in data['description']]
    data.payout_type = data.payout_type.replace('CHECK',1).replace('ACH',2)\
        .replace('',0)
    data.previous_payouts = [len(i) for i in data.previous_payouts]
    
    temp_array = data.apply(lambda row: calc_price_and_tickets(row['ticket_types']),axis=1)
    data['avg_cost']=split_tuple(temp_array, 'avg_cost')
    data['total_tickets']=split_tuple(temp_array, 'total_tickets')
    data.ticket_types = [len(i) for i in data.ticket_types]
    
    data.delivery_method = data.delivery_method.fillna(0).astype(int)
    data.has_header = data.has_header.fillna(0).astype(int)
    data.listed = data.listed.replace('y',1).replace('n',0)
    data.country = data.country.fillna('NotDefined')
    data.venue_country = data.venue_country.fillna('NotDefined').replace('','NotDefined')
    data.venue_state = data.venue_state.fillna('NotDefined').replace('','NotDefined')
    data.venue_name = data.venue_name.fillna('NotDefined').replace('','NotDefined')
    data['has_venue_address'] = [1 if x != '' else 0 for x in data.venue_address]
    data['country_match'] = data['country'] == data['venue_country']
    
    #process time data
    data = to_datetime(data, ['event_end','event_start', 'event_created', 'event_published'])
    data['event_duration_days']= data.apply(lambda row: calc_duration(row['event_start'], row['event_end']),axis=1)
    data['days_to_publish']= data.apply(lambda row: calc_duration(row['event_created'], row['event_published']),axis=1)
    data['days_until_event']= data.apply(lambda row: calc_duration(row['event_created'], row['event_start']),axis=1)
    
    #target
    data['fraud'] = [1 if 'fraud' in x else 0 for x in data.acct_type]
    data['not_fraud'] = [1 if not 'fraud' in x else 0 for x in data.acct_type]

    #Featurize text fields
    data, kmeans_description = text_to_cluster(data, 'description', 20)
    data, kmeans_venue_name = text_to_cluster(data, 'venue_name', 20)
    data, kmeans_name = text_to_cluster(data, 'name', 20)
	
    #Drop
    data.drop(['num_order', 'num_payouts', 'approx_payout_date',
               'email_domain', 'event_created', 'event_end',
               'event_published', 'event_start', 'gts', 'name_length',
               'object_id', 'org_desc', 'org_facebook', 'org_name',
               'org_twitter', 'payee_name', 'sale_duration', 'sale_duration2',
               'user_created', 'venue_latitude', 'venue_longitude','not_fraud', 'acct_type',
               'country','venue_country', 'currency', 'venue_address', 'venue_state'],
              axis=1, inplace=True)
    

    return data, kmeans_description, kmeans_venue_name, kmeans_name

def calc_duration(event1,event2):
    '''calculates the nubmer of hours between two events
    args:
        df: pandas dataframe
        event1: first event
        event2: second event
    ''' 
    
    if (event1 != event1) or (event2 != event2): #check for missing time stamps
        return 0
    return int((event2-event1).days)


def to_datetime(df, col_list):
    '''converts colums to datetime objects
    args:
        df: pandas dataframe
        col_list: list of columns to convert
    '''
    for col in col_list:
        df[col]= pd.to_datetime(df[col], unit = 's')
    return df

def vectorize(text):
    '''vectorizes string vector using pretrained doc2vec model'''
    if text:
        return model.infer_vector(text.strip().split(" "))
    else:
        return [0 for x in range(300)]

def text_to_cluster(df, col, num_clusters=8):
    '''
    vectorizes text with doc to vec and does k-means clustering
    '''
    df[col] = list(df[col].apply(lambda x: vectorize(x)))
    vectors= np.stack(df[col].values)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(vectors)
    df[col] = kmeans.labels_
    
    #order clusters by %Fraud
    df2=df.groupby([col]).sum().reset_index()
    df2['percent_fraud']=df2['fraud']/(df2['fraud']+df2['not_fraud'])
    cluster_indx = np.argsort(df2['percent_fraud'])
    sorting_dict ={v:k for k, v in zip(cluster_indx.index, cluster_indx)}
    df[col] = df.apply(lambda row: sorting_dict[row[col]],axis=1)
    return df, kmeans

def calc_price_and_tickets(ticket_type):
    '''calculate the average price for a ticket
    '''
    if len(ticket_type)>=1:
        cost=[]
        ticket_avail=[]
        for i in range(len(ticket_type)):
            cost.append(ticket_type[i]['cost'])
            ticket_avail.append(ticket_type[i]['quantity_total'])
        cost=np.array(cost)
        ticket_avail=np.array(ticket_avail)
        
        if np.sum(ticket_avail)>=1:
            avg_cost = (np.sum(cost*ticket_avail))/np.sum(ticket_avail)
        else:
            avg_cost = 0
            
        total_tickets = np.sum(ticket_avail)
        
        return avg_cost,total_tickets

def split_tuple(combined_array, param='avg_cost'):
    requested_list=[]
    if param == 'avg_cost':
        col = 0
    else:
        col = 1
    
    for i in range(len(combined_array)):
        if combined_array[i]:
            requested_list.append(combined_array[i][col])
        else:
            requested_list.append(0)
        
    return requested_list
        

if __name__ == '__main__':
    data = pd.read_json('data/data.json')
    data = clean(data)
