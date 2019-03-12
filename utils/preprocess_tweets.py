import json
import spacy
from bs4 import BeautifulSoup
import re
from collections import OrderedDict

path_to_data = "/home/chase/data/trump-tweets/trumptweets"
path_to_otpt = "/home/chase/data/trump-tweets/tweets.txt"

''' Preprocesses the original Twitter json data into one file
where each line is the text of a tweet. Retweets are skipped
entirely.
'''

with open(path_to_data, "r") as f:
    data=f.read()

parsed_json = json.loads(data, object_pairs_hook=OrderedDict)
text = []
count = 0

for i in range(len(parsed_json)):
    tweet = parsed_json[i]
    count+=1
    try:
        if not tweet['is_retweet']:
            # Remove URLs from tweets.
            # re taken from https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
            text_no_urls = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                    '', tweet['text'].strip())
            text_no_urls = re.sub('\.\.\.+', '', text_no_urls)
            # Tried to replace html escaped chars like this but it didn't work. Ending
            # up hardcoding &amp; but obviously this won't fix others is they exist.
            #html_decoded_string = str(BeautifulSoup(text_no_urls, "html.parser"))
            #html_decoded_string = html_decoded_string.replace('&amp;', '&')
            html_decoded_string = text_no_urls.replace('&amp;', '&')

            text.append(html_decoded_string)
    except:
        continue
text = [" ".join(text[::-1])]
print("Processed {} of {} tweets".format(len(text), count))
with open(path_to_otpt, "w") as o:
    for txt in text:
        o.write("{}\n".format(txt))
