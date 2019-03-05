import json

path_to_data = "/home/cddunca2/data/trump-tweets/trumptweets"
path_to_otpt = "/home/cddunca2/data/trump-tweets/tweets.txt"

''' Preprocesses the original Twitter json data into one file
where each line is the text of a tweet. Retweets are skipped
entirely.
'''

with open(path_to_data, "r") as f:
    data=f.read()

parsed_json = json.loads(data)
text = []
count = 0
for i in range(len(parsed_json)):
    tweet = parsed_json[i]
    count+=1
    try:
        if not tweet['is_retweet']:
            text.append(tweet['text'].strip())
    except:
        continue
print("Processed {} of {} tweets".format(len(text), count))
with open(path_to_otpt, "w") as o:
    for txt in text:
        o.write("{}\n".format(txt))
