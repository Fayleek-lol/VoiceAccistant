
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import random
from googlesearch import search  # ����� � Google
from termcolor import colored  # ����� ������� ����� (��� ��������� ������������ ����)
import speech_recognition  # ������������� ���������������� ���� (Speech-To-Text)
import pyttsx3  # ������ ���� (Text-To-Speech)
import webbrowser  # ������ � �������������� �������� �� ��������� (���������� ������� � web-���������)
import traceback  # ����� traceback ��� ��������� ������ ��������� ��� ������ ����������
import json  # ������ � json-������� � json-��������
import os  # ������ � �������� ��������


class Translation:
    """
    ��������� ������� � ���������� �������� ����� ��� �������� �������������� ����������
    """
    with open("translations.json", "r", encoding="UTF-8") as file:
        translations = json.load(file)

    def get(self, text: str):
        """
        ��������� �������� ������ �� ����� �� ������ ���� (�� ��� ����)
        :param text: �����, ������� ��������� ���������
        :return: ������ � ���������� ������� ������
        """
        if text in self.translations:
            return self.translations[text][assistant.speech_language]
        else:
            # � ������ ���������� �������� ���������� ����� ��������� �� ���� � ����� � ������� ��������� ������
            print(colored("Not translated phrase: {}".format(text), "red"))
            return text


class OwnerPerson:
    """
    ���������� � ���������, ���������� ���, ����� ����������, ������ ���� ����, ��������� ���� (��� ��������� ������)
    """
    name = ""
    native_language = ""
    target_language = ""


class VoiceAssistant:
    """
    ��������� ���������� ����������, ���������� ���, ���, ���� ����
    ����������: ��� ������������� ��������� ����������� ����� ������� ��������� �����,
    ������� ����� ����� ������� �� JSON-����� � ������ ������
    """
    name = ""
    sex = ""
    speech_language = ""
    recognition_language = ""


def setup_assistant_voice():
    """
    ��������� ������ �� ��������� (������ ����� �������� � ����������� �� �������� ������������ �������)
    """
    voices = ttsEngine.getProperty("voices")

    if assistant.speech_language == "en":
        assistant.recognition_language = "en-US"
        if assistant.sex == "female":
            # Microsoft Zira Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[1].id)
        else:
            # Microsoft David Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[2].id)
    else:
        assistant.recognition_language = "ru-RU"
        # Microsoft Irina Desktop - Russian
        ttsEngine.setProperty("voice", voices[0].id)


def record_and_recognize_audio(*args: tuple):
    """
    ������ � ������������� �����
    """
    with microphone:
        recognized_data = ""

        # ����������� ����� ��������� ��� ����������� ������� ����� �� ���
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            play_voice_assistant_speech(translator.get("Can you check if your microphone is on, please?"))
            traceback.print_exc()
            return

        # ������������� online-������������� ����� Google (������� �������� �������������)
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language=assistant.recognition_language).lower()

        except speech_recognition.UnknownValueError:
            pass

        return recognized_data


def play_voice_assistant_speech(text_to_speech):
    """
    ������������ ���� ������� ���������� ���������� (��� ���������� �����)
    :param text_to_speech: �����, ������� ����� ������������� � ����
    """
    ttsEngine.say(str(text_to_speech))
    ttsEngine.runAndWait()


def play_failure_phrase(*args: tuple):
    """
    ������������ ��������� ����� ��� ��������� �������������
    """
    failure_phrases = [
        translator.get("Can you repeat, please?"),
        translator.get("What did you say again?")
    ]
    play_voice_assistant_speech(failure_phrases[random.randint(0, len(failure_phrases) - 1)])


def play_greetings(*args: tuple):
    """
    ������������ ��������� �������������� ����
    """
    greetings = [
        translator.get("Hello, {}! How can I help you today?").format(person.name),
        translator.get("Good day to you {}! How can I help you today?").format(person.name)
    ]
    play_voice_assistant_speech(greetings[random.randint(0, len(greetings) - 1)])


def play_farewell_and_quit(*args: tuple):
    """
    ������������ ������������ ���� � �����
    """
    farewells = [
        translator.get("Goodbye, {}! Have a nice day!").format(person.name),
        translator.get("See you soon, {}!").format(person.name)
    ]
    play_voice_assistant_speech(farewells[random.randint(0, len(farewells) - 1)])
    ttsEngine.stop()
    quit()


def search_for_term_on_google(*args: tuple):
    """
    ����� � Google � �������������� ��������� ������ (�� ������ ����������� � �� ���� ����������, ���� ��������)
    :param args: ����� ���������� �������
    """
    search_term = " "

    # �������� ������ �� ��������� � ��������
    url = "https://google.com/search?q=" + search_term
    webbrowser.get().open(url)

    # �������������� ����� � �������������� ��������� ������ �� ���������� (� ��������� ������� ����� ���� �����������)
    search_results = []
    try:
        for _ in search(search_term,  # ��� ������
                        tld="com",  # ��������������� �����
                        lang=assistant.speech_language,  # ������������ ����, �� ������� ������� ���������
                        num=1,  # ���������� ����������� �� ��������
                        start=0,  # ������ ������� ������������ ����������
                        stop=1,  # ������ ���������� ������������ ���������� (� ����, ����� ���������� ������ ���������)
                        pause=1.0,  # �������� ����� HTTP-���������
                        ):
            search_results.append(_)
            webbrowser.get().open(_)

    # ��������� ��� ������ ����������� ������, �� ����� ���������� ����� � ����������� ������� ��� ��������� ���������
    except:
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    print(search_results)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on google").format(search_term))


def change_language(*args: tuple):
    """
    ��������� ����� ���������� ���������� (����� ������������� ����)
    """
    assistant.speech_language = "ru" if assistant.speech_language == "en" else "en"
    setup_assistant_voice()
    print(colored("Language switched to " + assistant.speech_language, "cyan"))


def toss_coin(*args: tuple):
    """
    "�������������" ������� ��� ������ �� 2 �����
    """
    flips_count, heads, tails = 3, 0, 0

    for flip in range(flips_count):
        if random.randint(0, 1) == 0:
            heads += 1

    tails = flips_count - heads
    winner = "Tails" if tails > heads else "Heads"
    play_voice_assistant_speech(translator.get(winner) + " " + translator.get("won"))


# �������� ������ ��� ������������� � ���� JSON-�������
config = {
    "intents": {
        "greeting": {
            "examples": ["������", "����������", "������ ����",
                         "hello", "good morning"],
            "responses": play_greetings
        },
        "farewell": {
            "examples": ["����", "�� ��������", "��������", "�� �������",
                         "goodbye", "bye", "see you soon"],
            "responses": play_farewell_and_quit
        },
        "google_search": {
            "examples": ["����� � ����",
                         "search on google", "google", "find on google"],
            "responses": search_for_term_on_google
        },
        "language": {
            "examples": ["����� ����", "������� ����",
                         "change speech language", "language"],
            "responses": change_language
        },
        "toss_coin": {
            "examples": ["�������� �������", "������� �������",
                         "toss coin", "coin", "flip a coin"],
            "responses": toss_coin
        }
    },

    "failure_phrases": play_failure_phrase
}


def prepare_corpus():
    """
    ���������� ������ ��� ���������� ��������� ������������
    """
    corpus = []
    target_vector = []
    for intent_name, intent_data in config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            target_vector.append(intent_name)

    training_vector = vectorizer.fit_transform(corpus)
    classifier_probability.fit(training_vector, target_vector)
    classifier.fit(training_vector, target_vector)


def get_intent(request):
    """
    ��������� �������� ���������� ��������� � ����������� �� ������� ������������
    :param request: ������ ������������
    :return: �������� ��������� ���������
    """
    best_intent = classifier.predict(vectorizer.transform([request]))[0]

    index_of_best_intent = list(classifier_probability.classes_).index(best_intent)
    probabilities = classifier_probability.predict_proba(vectorizer.transform([request]))[0]

    best_intent_probability = probabilities[index_of_best_intent]

    # ��� ���������� ����� ��������� ����� ��������� ���� ����������
    print(best_intent_probability)
    if best_intent_probability > 0.157:
        return best_intent


def make_preparations():
    """
    ���������� ���������� ���������� � ������� ����������
    """
    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, \
        classifier

    # ������������� ������������ ������������� � ����� ����
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    # ������������� ����������� ������� ����
    ttsEngine = pyttsx3.init()

    # ��������� ������ ������������
    person = OwnerPerson()
    person.name = "����"
    person.native_language = "ru"
    person.target_language = "en"

    # ��������� ������ ���������� ���������
    assistant = VoiceAssistant()
    assistant.name = "Rob"
    assistant.sex = "male"
    assistant.speech_language = "en"

    # ��������� ������ �� ���������
    setup_assistant_voice()

    # ���������� ������������ �������� ���� (�� �������������� �����)
    translator = Translation()

    # ���������� ������� ��� ������������� �������� ������������ � ��������� ������������ (����� �������)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()


if __name__ == "__main__":
    make_preparations()

    while True:
        # ����� ������ ���� � ����������� ������� ������������ ���� � ��������� ����������� � �������� �����
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        # ��������� ������� �� �������������� ���������� (����������)
        if voice_input:
            voice_input_parts = voice_input.split(" ")

            # ���� ���� ������� ���� ����� - ��������� ������� ����� ��� �������������� ����������
            if len(voice_input_parts) == 1:
                intent = get_intent(voice_input)
                if intent:
                    config["intents"][intent]["responses"]()
                else:
                    config["failure_phrases"]()

            # � ������ ������� ����� - ����������� ����� �������� ����� � ���������� ����� ������ �����,
            # ���� �� ����� ������� ����������
            if len(voice_input_parts) > 1:
                for guess in range(len(voice_input_parts)):
                    intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    print(intent)
                    if intent:
                        command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        print(command_options)
                        config["intents"][intent]["responses"](*command_options)
                        break
                    if not intent and guess == len(voice_input_parts)-1:
                        config["failure_phrases"]()
