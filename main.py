import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    global_settings = {}
    global_settings.update(configuration)
    print("Configuration updated:", global_settings)


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []

    questions = [
        "optics is the study of what?",
        "what does dna stand for?",
        "which apollo moon mission was the first to carry a lunar rover?",
        "what was the name of the first man-made satellite launched by the soviet union in 1957?",
        "what is the rarest blood type?",
        "the earth has three layers that are different due to varying temperatures. what are its three layers?",
        "frogs belong to which animal group?",
        "how many bones do sharks have in their bodies?",
        "the smallest bones in the body are located where?",
    ]
    answers = [
        "Light",
        "Deoxyribonucleic Acid",
        "Apollo 15 mission",
        "Sputnik 1",
        "AB Negative",
        "Crust, mantle, and core",
        "Amphibians",
        "Zero!",
        "The ear",
    ]

    for text in request.text:
        user_question = text.lower()
        user_question = nltk.word_tokenize(user_question)

        stop_words = set(stopwords.words("english"))
        user_question = [word for word in user_question if word not in stop_words]

        questions.append(user_question)

        vectorizer = TfidfVectorizer().fit_transform(questions)

        cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])

        most_similar_question_index = cosine_similarities.argmax()

        response = answers[most_similar_question_index]

        output.append(response)

    return SimpleText(dict(text=output))
