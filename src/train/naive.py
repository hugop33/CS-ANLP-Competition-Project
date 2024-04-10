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

        "technolo": "Technology",
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
    print(label_words)

    test_df = pd.DataFrame(test_sentences, columns=['sentence'])
    for label, words in label_words.items():
        test_df[label] = test_df['sentence'].apply(lambda x: len([word for word in words if word in x]))
    test_df["Total"] = test_df[label_words.keys()].sum(axis=1)
    test_df["Labels"] = test_df.apply(lambda x: [label for label in label_words.keys() if x[label] > 0], axis=1)
    test_df["Label"] = test_df["Labels"].apply(lambda x: x[0] if len(x)==1 else "Others")
    test_df.drop(columns=[col for col in label_words.keys()]+["Labels", "Total", "sentence"], inplace=True)
    test_df["ID"] = test_df.index
    print(test_df.head())
    test_df.to_csv('./data/partial_naive.csv', index=False)

if __name__=="__main__":
    classify()