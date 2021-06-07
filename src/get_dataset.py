from bs4 import BeautifulSoup
from collections import defaultdict
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import unquote
from selenium.webdriver.chrome.options import Options
from selenium import webdriver 
from html.parser import HTMLParser

from datetime import timedelta, date
import requests
import json
import pandas as pd 
import time
from tqdm import tqdm
from ast import literal_eval
import re
from collections import defaultdict
from copy import deepcopy
import os
import argparse

h = HTMLParser()
parser = argparse.ArgumentParser(description="SOCCER Dataset")
parser.add_argument('-dp', '--driver_path', help = 'path of chromedriver', required=True, type=str)

args = vars(parser.parse_args())
driver_path = args['driver_path']

###############################################################
###### Getting URLs ##########################################
###############################################################

def get_urls(url, filepath):
    r = requests.get(url, allow_redirects=True)
    with open(filepath, 'wb') as f:
        f.write(r.content)

###############################################################
###### DATA CRAWLING ##########################################
###############################################################

def get_lineup(match_url, driver_path):
    driver = webdriver.Chrome(executable_path = driver_path)
    example_url = "/lineups/".join(match_url.rsplit("/",1))
    driver.get("https://"+example_url)
    com_page = BeautifulSoup(driver.page_source,"html.parser")
    match_score = com_page.find_all('div',{'class':'widget-match-header__score'})[0].text
    # getting team 1 lineup list
    
    def get_sub_lineup(div_name):
        lineup_block = com_page.find_all('ul',{'class':div_name})
        team_list = []
        for team in lineup_block:
            players = []
            names = team.find_all('span', {'class':'widget-match-lineups__name'})
            numbers = team.find_all('span', {'class':'widget-match-lineups__number'})
            final_names = [item.text for item in names]
            final_numbers = [item.text for item in numbers]
            for i in range(len(final_numbers)):
                players.append([final_numbers[i],final_names[i]])
            team_list.append(players)
        return team_list[0], team_list[1]
    
    lineup_name = 'widget-match-lineups__list widget-match-lineups__starting-eleven'
    team_home_lineup, team_away_lineup = get_sub_lineup(lineup_name)
    
    subs_name = 'widget-match-lineups__list widget-match-lineups__substitute'
    team_home_sub, team_away_sub = get_sub_lineup(subs_name)
    
    coaches = [item.text.strip() for item in com_page.find_all('a',{'class': 'widget-match-lineups__manager'})]
    driver.quit()
    return [[team_home_lineup, team_home_sub],[team_away_lineup, team_away_sub],coaches]

def get_commentary_events(match_url, driver_path):
    driver = webdriver.Chrome(executable_path = driver_path)
    example_url = "/commentary-result/".join(match_url.rsplit("/",1))
    driver.get("https://"+example_url)
    btn_xpath = './/div[@class="content-collapser content-collapser--btn"]'
    event_btn_xpath = './/div[@class="btn btn--outline"]'
    comments = []
    events = []
    # get the commentary of the game
    try:
        WebDriverWait(driver, 3).until(EC.visibility_of_element_located((By.XPATH, btn_xpath)))
        element = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, btn_xpath)))
        print("Done waiting - commentary")
        ele = driver.find_element_by_xpath(btn_xpath)#.click()
        driver.execute_script("arguments[0].click();", ele)
        com_page = BeautifulSoup(driver.page_source,"html.parser")
        comments_block = com_page.find_all('div',{'class':'comment'})
        
        if comments_block != []:
            for idx, comment in enumerate(comments_block):
                comments.append(comment.text)
        print("Commentary done appending")
    except:
        print("Fail to crawl/No commentary available")
        
    # get the key events of the game
    try:
        WebDriverWait(driver, 3).until(EC.visibility_of_element_located((By.XPATH, event_btn_xpath)))
        element = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, event_btn_xpath)))
        print("Done waiting - events")
        ele = driver.find_element_by_xpath(event_btn_xpath)#.click()
        driver.execute_script("arguments[0].click();", ele)
        event_page = BeautifulSoup(driver.page_source,"html.parser")
        key_events = event_page.find('div',{'class':'widget-match-key-events in-drawer'}).find_all('div',{'class':'event'})
        
        def get_events_per_team(div_name, events):
                if div_name == "event-text team-home clearfix":
                    team_name = "Home Team"
                else:
                    team_name = "Away Team"
                if event.find_all("div",{"class":div_name}):
                    time = event.find("div", {'class':'event-time'}).text.strip()
                    player = event.find("div", {'class':'event-text-main'}).text.strip()
                    additional = event.find("div", {'class':'event-text-additional'}).text.strip()
                    if event.find("div", {"class": 'match-event-icon type-goal'}):
                        action = "Goal"
                        if additional != "Goal":
                            events.append([team_name, time, "Assist", additional, ""])
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": 'match-event-icon type-yellow_card'}):
                        action = "Yellow Card"
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": 'match-event-icon type-substitution'}):
                        action = "Switch"
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": "match-event-icon type-penalty_goal"}):
                        action = "Penalty Goal"
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": "match-event-icon type-second_yellow_card"}):
                        action = "Yellow 2nd/RC"
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": "match-event-icon type-own_goal"}):
                        action = "Own Goal"
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    elif event.find("div", {"class": "match-event-icon type-red_card"}):
                        action = "Red Card"
                        additional = ""
                        events.append([team_name, time, action, player, additional])
                    else:
                        pass

                    
                return events
                        
        if key_events != []:
            for idx, event in enumerate(key_events):
                home_div = "event-text team-home clearfix"
                away_div = "event-text team-away clearfix"
                events = get_events_per_team(home_div, events)
                events = get_events_per_team(away_div, events)
        print("Events done appending")
    except:
        print("Fail to crawl/No events available")
        
    driver.quit()
    reversed_comments = comments[::-1]
    reversed_events = events[::-1]
    
    return reversed_comments, reversed_events


###############################################################
###### FOR COMMENT FORMATTING #################################
###############################################################

def process_commentary(comment_list):
    
    processed_comment = []
    for i in range(len(comment_list)):
        comment_list[i] = h.unescape(comment_list[i]).replace('\xa0', ' ')
        token_list = comment_list[i].strip().split("      ")
        token_list = [tokens.strip() for tokens in token_list if tokens != ""]
        
        if len(token_list) > 1:
            if token_list[-2][0].isdigit():
                token_list[-1] = "  ".join([token_list[-2],token_list[-1]])

        comment_cleaned = token_list[-1]
        comment_separated = comment_cleaned.strip().split("  ")
#         the first element could be team scores
        token_length = len(comment_separated)
        if token_length == 1:
            processed_comment.append(['BREAK',comment_separated[0]])
        elif token_length >= 2:
            if len(comment_separated[0]) > 10:
                processed_comment.append(['BREAK'," ".join(comment_separated[1:])])
            else:
                processed_comment.append([comment_separated[0]," ".join(comment_separated[1:])])
        else:
            print("Wrong Formats in commentary, possible 0-length sentences")
    
    return processed_comment

def filter_score(processed_comment):
    # roughly removed all score-revealing sentences
    filtered_comment = []
    for i in tqdm(processed_comment):
        comment = i[1]
        pattern = re.sub("([A-Z]*\s*[A-Z]*\s*[A-Z]+)\s*([0-9]+) ?- ?([0-9]+)\s*([A-Z]+\s*[A-Z]*\s*[A-Z]*\s*)\s*\W*","",comment)
        if len(pattern) > 15 or pattern.isupper() != True:
            filtered_comment.append([i[0],pattern])
    return filtered_comment

def change_label(processed_comment):
    # change the labels to have more accurate timestamps such as START and END
    num_comment = len(processed_comment)
    timestamps = []
    started = False
    for idx in range(len(processed_comment)):
        timestamp = processed_comment[idx][0]
        if timestamp.isalpha() and started == False:
            timestamp = 'START'
            processed_comment[idx][0] = timestamp
        elif timestamp.isalpha() == False and started == False:
            started = True
            processed_comment[idx][0] = timestamp
        elif timestamp.isalpha() and started == True:
            if idx+5 < num_comment-1:
                if processed_comment[idx+5][0].isalpha() == False:
                    processed_comment[idx][0] = timestamp
                else:
                    timestamp = 'END'
                    processed_comment[idx][0] = timestamp
            else:
                timestamp = 'END'
                processed_comment[idx][0] = timestamp
        else:
            processed_comment[idx][0] = timestamp

    return processed_comment

def format_commentary(comment):
    timestamps = [c[0].strip() for c in comment]
    formatted_comments = {time:[] for time in timestamps} 
    for idx in range(len(comment)):
        time = comment[idx][0].strip()
        c = comment[idx][1]
        formatted_comments[time].extend([c])
    final_dic = {key:" ".join(value) for key,value in formatted_comments.items()}    
    return final_dic  

###############################################################
###### FOR EVENTS FORMATTING ##################################
###############################################################

def create_event():
    event_keys = ['goal','assist','swap','yellow_card','red_card']
    team_event = {k:[] for k in event_keys}
    events = {"home":deepcopy(team_event), "guest":deepcopy(team_event),"cumulative_score":[0,0]}
    return events

def update_team_event(event, time_dict):
    guest_event = time_dict['guest']
    home_event = time_dict["home"]
    if event[0] == 'Away Team':
        if event[2] == 'Own Goal' or event[2] == 'Goal' or event[2] == 'Penalty Goal' :
            guest_event['goal'].append(event[3])
        elif event[2] == 'Assist':
            guest_event['assist'].append(event[3])
        elif event[2] == 'Yellow Card':
            guest_event['yellow_card'].append(event[3])
        elif event[2] == 'Yellow 2nd/RC' or event[2] == 'Red Card':
            guest_event['red_card'].append(event[3])
        else:
            guest_event['swap'].append([event[3],event[4]])
    elif event[0] == 'Home Team':
        if event[2] == 'Own Goal' or event[2] == 'Goal' or event[2] == 'Penalty Goal' :
            home_event['goal'].append(event[3])
        elif event[2] == 'Assist':
            home_event['assist'].append(event[3])
        elif event[2] == 'Yellow Card':
            home_event['yellow_card'].append(event[3])
        elif event[2] == 'Yellow 2nd/RC' or event[2] == 'Red Card':
            home_event['red_card'].append(event[3])
        else:
            home_event['swap'].append([event[3],event[4]])       
    return time_dict

def format_events(events):
# filtering out anomalies
    events = [i for i in events if len(i) == 5]
    time_keys = [e[1] for e in events]
    all_state = {time:deepcopy(create_event()) for time in time_keys}
    
    for idx in range(len(events)):
        event = events[idx]
        time = events[idx][1]
        time_dict = all_state[time]
        time_dict = update_team_event(event,time_dict)
        if idx == 0:
            time_dict['cumulative_score'] = [0,0]
#           away team has an own goal -> home team + 1
            if event[0] == 'Away Team' and event[2] == 'Own Goal':
                time_dict['cumulative_score'][0] += 1
#           away team has a goal/penalty goal -> away team + 1                                            
            elif event[0] == 'Away Team' and (event[2] == 'Goal' or event[2] == 'Penalty Goal'):
                time_dict['cumulative_score'][1] += 1
#           home team has an own goal -> away team + 1
            elif event[0] == 'Home Team' and event[2] == 'Own Goal' :
                time_dict['cumulative_score'][1] += 1
#           home team has a goal/penalty goal -> home team + 1       
            elif event[0] == 'Home Team' and (event[2] == 'Goal' or event[2] == 'Penalty Goal'):
                time_dict['cumulative_score'][0] += 1
            
        else:
            previous_time = events[idx-1][1]
            previous_score = deepcopy(all_state[previous_time]['cumulative_score'])
            time_dict['cumulative_score'] = previous_score
            if event[0] == 'Away Team' and (event[2] == 'Goal' or event[2] == 'Penalty Goal'):
                time_dict['cumulative_score'][1] += 1
            elif event[0] == 'Away Team' and event[2] == 'Own Goal':
                time_dict['cumulative_score'][0]+= 1
            elif event[0] == 'Home Team' and event[2] == 'Own Goal':
                time_dict['cumulative_score'][1] += 1
            elif event[0] == 'Home Team' and (event[2] == 'Goal' or event[2] == 'Penalty Goal'):
                time_dict['cumulative_score'][0] += 1
    return all_state

###############################################################
###### FOR LINEUP FORMATTING ##################################
###############################################################

def format_lineup(lineup):
    home_lineup, home_subs = lineup[0][0], lineup[0][1]
    guest_lineup, guest_subs = lineup[1][0], lineup[1][1]
    home_coach, guest_coach = lineup[-1][0], lineup[-1][1]
    return home_lineup, home_subs, home_coach, guest_lineup, guest_subs, guest_coach

###############################################################
###### GENERAL PIPELINE & FORMATTING ##########################
###############################################################

def general_formatting(df):
    idx = range(len(df))
    final_dic = {i:{} for i in idx}
    df['team_1'] = df.url.apply(lambda row: row.rsplit("/",2)[1].split('-v-')[0]).apply(unquote)
    df['team_2'] = df.url.apply(lambda row: row.rsplit("/",2)[1].split('-v-')[1]).apply(unquote)
    for i in idx:
        final_dic[i] = {
            'url': df.url.iloc[i],
            'date': df.date.iloc[i],
            'comp_title':df.comp_title.iloc[i],
            'home':df.team_1.iloc[i],
            'guest':df.team_2.iloc[i],
            'score':df.score.iloc[i].strip(),
            'home_lineup':df.formatted_lineups.iloc[i][0],
            'home_subs':df.formatted_lineups.iloc[i][1],
            'home_coach':df.formatted_lineups.iloc[i][2],
            'guest_lineup':df.formatted_lineups.iloc[i][3],
            'guest_subs':df.formatted_lineups.iloc[i][4],
            'guest_coach':df.formatted_lineups.iloc[i][5],
            'commentary':df.formatted_comments.iloc[i],
            'states':df.formatted_states.iloc[i]
        }
    return final_dic
        
def get_dataset(datafile, outfile):
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    df = pd.read_csv(datafile)
    if outfile == 'data/train.json':
        imfile = './temp/train_im.csv'
    elif outfile == 'data/val.json':
        imfile = './temp/val_im.csv'
    elif outfile == 'data/test.json':
        imfile = './temp/test_im.csv'
    else:
        imfile = './temp/check.csv'
        
    if os.path.isfile(imfile):
        im_df = pd.read_csv(imfile)
        im_df['comments'] = im_df['comments'].apply(literal_eval)
        im_df['events'] = im_df['events'].apply(literal_eval)
        im_df['lineups'] = im_df['lineups'].apply(literal_eval)
    else:
        im_df = pd.DataFrame(columns=['url','comments','events','lineups'])
        
    next_idx = len(im_df)
    
#     if the intermediate file is the same length as the url, delete the intermediate file
    if next_idx == len(df):
        # os.remove(imfile)
        pass
    else:
        urls = list(df.url.iloc[next_idx:])
        for index, url in enumerate(tqdm(urls)):
            c, e = get_commentary_events(url, driver_path)
            lineup = get_lineup(url, driver_path)
            new_df = pd.DataFrame([[url, c, e, lineup]], columns=['url','comments','events','lineups'])
            im_df = im_df.append(new_df,ignore_index=True)
            if index % 5 == 0:
                im_df.to_csv(imfile,index=None)
            elif (len(urls) - index) < 5:
                im_df.to_csv(imfile,index=None)
   
    
    df['comments'] = im_df.comments
    df['events'] = im_df.events
    df['lineups'] = im_df.lineups
    df['formatted_comments'] = df.comments.apply(process_commentary).apply(filter_score).apply(change_label).apply(format_commentary)
    df['formatted_states'] = df.events.apply(format_events)
    df['formatted_lineups'] = df.lineups.apply(format_lineup)
    dataset = general_formatting(df)
    with open(outfile, 'w') as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    train_url = 'https://www.dropbox.com/s/u4blltjqwp77eco/train_urls.csv?dl=1'
    val_url = 'https://www.dropbox.com/s/p9vz5y3xkmxqpvw/val_urls.csv?dl=1'
    test_url = 'https://www.dropbox.com/s/twa5u4p8otc8aud/test_urls.csv?dl=1'
    
    url_folder = "urls/"
    data_folder = "data/"
    
    if not os.path.exists(url_folder):
        os.mkdir(url_folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)        
    
    train_file = url_folder + "train_url.csv"
    val_file = url_folder + "val_url.csv"
    test_file = url_folder + "test_url.csv"
    
    get_urls(train_url, train_file)
    get_urls(val_url, val_file)
    get_urls(test_url, test_file)
        
    train_out = data_folder + "train.json"
    val_out = data_folder + "val.json"
    test_out = data_folder + "test.json"
    
    get_dataset(train_file, train_out)
    get_dataset(val_file, val_out)
    get_dataset(test_file, test_out)
    