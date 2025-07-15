import re
import sys
import os

# sys.path.append('/home/alisa/Documents/GitHubProjects/WhatIsPErsuasiveTextFromLLMsPointOfView')


def extract_features(doc):
    """Extract storytelling features like metaphors and narrative indicators."""
    storytelling_features = {'narrative_words': 0, 'comparison_count': 0}
    narrative_indicators = { 'alors','bon', 'donc', 'enfin', 'quoi', 'voila', 'il était une fois',  'raconter',  'histoire',  'conte',  'un jour',  'dans un pays lointain',  'il y a longtemps', 
                             'autrefois',  'c’est alors que',  'soudain',   'et puis',  'ensuite',   'tout à coup',  'ce fut alors',  'il advint que',  
                             'par la suite',   'au début',  'finalement',  'enfin',  'ainsi','comme dans un rêve',
                             'dans cette époque lointaine','selon la légende','on raconte que'}
    


    comparison_indicators = {'comme', 'tel', 'pareil à', 'semblable à', 'ressemble à', 'on dirait', 'c’est une sorte de', 'figure de', 'en forme de',
                            'à l’image de', 'emblème de', 'symbole de', 'allégorie de', 'métaphore de', 'image de', 'au sens figuré', 'pris pour', 
                            'pris comme', 'dans un sens élargi', 'dans un autre sens', 'à la manière de',  'à l’instar de'}
    for sentence in re.split(r'(?<=[.!?])\s+', doc.text):
        if any(phrase in sentence.lower() for phrase in narrative_indicators):
            storytelling_features['narrative_words'] += 1
        if any(phrase in sentence.lower() for phrase in comparison_indicators):
            storytelling_features['comparison_count'] += 1
    return storytelling_features
    
    
# Would be great to include this:
#Hypophora: Figure of reasoning in which one or more questions is/are asked and then answered, often at length, by one and the same speaker; raising and responding to one's own question(s).

#Examples

#"When the enemy struck on that June day of 1950, what did America do? It did what it always has done in all its times of peril. It appealed to the heroism of its youth."
