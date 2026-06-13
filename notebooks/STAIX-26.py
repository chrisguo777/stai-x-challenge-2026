#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:57:15 2026

@author: hasnaghamdi
"""
#import library 
import pandas as pd


# Read data set:
valcov= pd.read_csv('/Users/hasnaghamdi/Downloads/covariates_v.csv')
traincov= pd.read_csv('/Users/hasnaghamdi/Downloads/covariates_t.csv')
sys_train= pd.read_csv('/Users/hasnaghamdi/Downloads/dose_sys_train.csv')

# get the column names
valcov.head()
traincov.head()
sys_train.head()

# get how many rows/columns
valcov.shape
traincov.shape
sys_train.shape


#read mereged data:

tad=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/train_all_drugs_merged.csv")
tao=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/train_all_opioids_merged.csv")
tas=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/train_all_stimulants_merged.csv")
tu=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/train_universal_merged.csv")
vad=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/val_all_drugs_merged.csv")
vao=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/val_all_opioids_merged.csv")
vas=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/val_all_stimulants_merged.csv")
vu=pd.read_csv("/Users/hasnaghamdi/Downloads/outputs/outputs/val_universal_merged.csv")

#couunt the common words

common_words= tu["state_doh_release"].fillna("").str.lower()

from collections import Counter
import re

all_words= []
for text in common_words:
    words = re.findall(r"\b[a-z]{3,}\b", text) 
    all_words.extend(words)
    
word_counts=Counter(all_words)

word_counts.most_common(100)

#cluster the risk based on the most repeated words to three classes:
crisis_words=[
    "overdose",
    "fentanyl",
    "overdose",
    "xylazine",
    "opioid"]

alert_words=[
    "suspected",
    "reported",
    "cluster",
    "emergency",
    "preliminary",
    "witnessing",
    "surveillance"]

action_words=[
    "treatment",
    "medication",
    "programs",
    "recovery",
    "services",
    "naloxone",
    "enforcement",
    "reduction"]

#create new columns for each class:

text = tu["state_doh_release"].fillna("").str.lower()
tu["crisis_score"] = text.str.count("|".join(crisis_words))
tu["alert_score"] = text.str.count("|".join(alert_words))
tu["action_score"] = text.str.count("|".join(action_words))

tu["adjustment"] = 0
tu.loc[text.str.contains("reported|data|surveillance", na=False), "crisis_score"] -= 1

def classify(row):
    scores = {
        "CRISIS": row["crisis_score"],
        "ALERT": row["alert_score"],
        "ACTION": row["action_score"]
    }

    max_class = max(scores, key=scores.get)

    if scores[max_class] <= 0:
        return "UNKNOWN"

    return max_class

tu["risk_classes"] = tu.apply(classify, axis=1)

tu["risk_classes"].value_counts()

#**** for the validiation data*******#


common_words= vu["state_doh_release"].fillna("").str.lower()

all_words= []
for text in common_words:
    words = re.findall(r"\b[a-z]{3,}\b", text) 
    all_words.extend(words)
    
word_counts=Counter(all_words)

word_counts.most_common(100)

crisis_words=[
    "overdose",
    "fentanyl",
    "overdose",
    "xylazine",
    "opioid"]

alert_words=[
    "suspected",
    "reported",
    "detected",
    "emergency",
    "preliminary",
    "witnessing",
    "surveillance",
    "wounds"]

action_words=[
    "treatment",
    "medication",
    "programs",
    "recovery",
    "services",
    "naloxone",
    "enforcement",
    "reduction"]

text = vu["state_doh_release"].fillna("").str.lower()
vu["crisis_score"] = text.str.count("|".join(crisis_words))
vu["alert_score"] = text.str.count("|".join(alert_words))
vu["action_score"] = text.str.count("|".join(action_words))

vu["adjustment"] = 0
vu.loc[text.str.contains("reported|data|surveillance", na=False), "crisis_score"] -= 1

def classify(row):
    scores = {
        "CRISIS": row["crisis_score"],
        "ALERT": row["alert_score"],
        "ACTION": row["action_score"]
    }

    max_class = max(scores, key=scores.get)

    if scores[max_class] <= 0:
        return "UNKNOWN"

    return max_class

vu["risk_classes"] = tu.apply(classify, axis=1)

vu["risk_classes"].value_counts()






