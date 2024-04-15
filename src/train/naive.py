import pandas as pd
import random

STAR_WORDS = {
        "politic": "Politics",
        "mayor": "Politics",
        "president": "Politics",
        "election": "Politics",
        "vote": "Politics",
        "government": "Politics",
        "parliament": "Politics",
        "minister": "Politics",

        "sport": "Sports",
        # "football": "Sports",
        # "ball": "Sports",
        "game": "Sports",
        "team": "Sports",
        "player": "Sports",

        "techno": "Technology",
        # "comput": "Technology",
        # "internet": "Technology",
        # "software": "Technology",
        # "app": "Technology",

        "busi": "Finance",
        "financ": "Finance",
        # "stock": "Finance",
        "market": "Finance",
        "econom": "Finance",
        # "bank": "Finance",
        # "invest": "Finance",
        # "fund": "Finance",
        # "money": "Finance",
        "compan": "Finance",

        "entertain": "Entertainment",
        "movi": "Entertainment",
        "music": "Entertainment",
        # "game": "Entertainment",
        "celebr": "Entertainment",
        "actor": "Entertainment",
        "actress": "Entertainment",
        "film": "Entertainment",
        "song": "Entertainment",
        # "play": "Entertainment",
        "drama": "Entertainment",
        "theater": "Entertainment",
        "concert": "Entertainment",

        "health": "Health",
        "medic": "Health",
        "doctor": "Health",
        "hospital": "Health",
        # "nurs": "Health",
        # "diseas": "Health",
        # "vaccin": "Health",
        # "covid": "Health",
        # "corona": "Health",

        "scienc": "Science",
        "research": "Science",
        "discover": "Science",
        # "scienti": "Science",
        # "experiment": "Science",
        "lab": "Science",
        "physic": "Science",
        # "chemi": "Science",
        # "biolog": "Science",
        # "math": "Science",

        "educat": "Education",
        "school": "Education",
        "student": "Education",
        # "teacher": "Education",
        # "univers": "Education",
        # "class": "Education",
        # "learn": "Education",
        # "lesson": "Education",
        # "book": "Education",
        # "exam": "Education",
        # "test": "Education",
        # "grade": "Education",

        "enviro": "Environment",
        "climat": "Environment",
        # "earth": "Environment",
        # "planet": "Environment",
        # "forest": "Environment",
        # "bio": "Environment",
        # "green": "Environment",
        # "pollut": "Environment",
        # "recycl": "Environment",
        # "sustain": "Environment",
        # "wast": "Environment",
        "water": "Environment",
        "ocean": "Environment",
        # "sea": "Environment",

        "food": "Food",
        # "cook": "Food",
        # "restaur": "Food",
        # "meal": "Food",
        # "eat": "Food",
        "drink": "Food",
        # "tast": "Food",
        # "flavor": "Food",
        # "cuisin": "Food",
        # "chef": "Food",
        # "bake": "Food",
        # "sweet": "Food",

        "travel": "Travel",
        # "tour": "Travel",
        # "trip": "Travel",
        # "vacat": "Travel",
        # "journey": "Travel",
        # "flight": "Travel",
        # "hotel": "Travel",
        # "resort": "Travel",
        # "beach": "Travel",
        "mountain": "Travel",
        # "adventur": "Travel",
        "country": "Travel",
        "foreign": "Travel",
        "passport": "Travel",

        "fashion": "Fashion",
        "style": "Fashion",
        "cloth": "Fashion",
        # "dress": "Fashion",
        # "design": "Fashion",
        # "brand": "Fashion",
        # "model": "Fashion",
        # "shoe": "Fashion",
        # "bag": "Fashion",
        # "jewel": "Fashion",
        "accessori": "Fashion",
        "makeup": "Fashion",
        "beauti": "Fashion",
    }

def classify():
    with open('./data/test_shuffle.txt') as f:
        test_sentences = f.readlines()
    
    label_words = {}
    for word, label in STAR_WORDS.items():
        if label not in label_words:
            label_words[label] = []
        label_words[label].append(word)
    # print(label_words)

    test_df = pd.DataFrame(test_sentences, columns=['sentence'])
    for label, words in label_words.items():
        test_df[label] = test_df['sentence'].apply(lambda x: len([word for word in words if word in x]))
    test_df["Total"] = test_df[label_words.keys()].sum(axis=1)
    test_df["Labels"] = test_df.apply(lambda x: [label for label in label_words.keys() if x[label] > 0], axis=1)
    test_df["Label"] = test_df["Labels"].apply(lambda x: x[0] if len(x)==1 else "Others")
    test_df.drop(columns=[col for col in label_words.keys()]+["Total", "sentence"], inplace=True)
    test_df["ID"] = test_df.index
    # print(test_df.head())
    test_df.to_csv('./data/partial_naive.csv', index=False)

if __name__=="__main__":
    classify()

    naive_df = pd.read_csv('./data/partial_naive.csv')
    deberta_df = pd.read_csv('./data/target_and_confidence.csv')
    deberta_df["Label1"] = deberta_df["Label1"].apply(lambda x: x.title())
    deberta_df["Label2"] = deberta_df["Label2"].apply(lambda x: x.title())

    thresh = 0.95
    merged_df = pd.merge(naive_df, deberta_df, on='ID')
    
    # print(merged_df.Confidence1.describe())
    # print(merged_df.head())
    # print number of rows where we have two naive labels in "Labels", et one of them is in the two labels Label1 and Label2
    merged_df["NbLabels"] = merged_df["Labels"].apply(lambda x: len(x.split(',')) if type(x)==str else 0)
    print(merged_df.head(10))
    print("nb others", len(merged_df[merged_df["Label"]=="Others"]))
    merged_df["L2_in_Labels"] = merged_df.apply(lambda x: x.NbLabels>1 and (x.Label1 not in x.Labels and x.Label2 in x.Labels), axis=1)
    merged_df["L1_in_Labels"] = merged_df.apply(lambda x: x.NbLabels>1 and (x.Label1 in x.Labels and x.Label2 not in x.Labels), axis=1)
    print(len(merged_df[merged_df["L2_in_Labels"]]))
    print(len(merged_df[merged_df["L1_in_Labels"]]))
    merged_df["Label"] = merged_df.apply(lambda x: x["Label2"] if x["L2_in_Labels"] else x["Label"], axis=1)
    print("nb others", len(merged_df[merged_df["Label"]=="Others"]))
    merged_df["Label"] = merged_df.apply(lambda x: x["Label1"] if x["L1_in_Labels"] else x["Label"], axis=1)

    print(len(merged_df[(merged_df["Confidence1"]<thresh) & (merged_df["Label"]!="Others")]))
    print("nb others", len(merged_df[merged_df["Label"]=="Others"]))
    # replace the last others with the most confident label
    merged_df["Label"] = merged_df.apply(lambda x: x["Label1"] if x["Label"]=="Others" else x["Label"], axis=1)
    print("nb others", len(merged_df[merged_df["Label"]=="Others"]))

    merged_df.drop(columns=["Label1", "Label2", "Confidence1", "Confidence2", "Labels", "L2_in_Labels", "L1_in_Labels", "NbLabels"], inplace=True)
    merged_df.to_csv('./data/merged_naive_deberta.csv', index=False)
    print(merged_df.head(10))