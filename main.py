from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from googlesearch import search
from termcolor import colored
from dotenv import load_dotenv
import speech_recognition
import googletrans
import pyttsx3
import random
import webbrowser
import traceback
import json
import os


class Translation:

    with open("translations.json", "r", encoding="UTF-8") as file:
        translations = json.load(file)

    def get(self, text: str):

        if text in self.translations:
            return self.translations[text][assistant.speech_language]
        else:
            # в случае отсутствия перевода происходит вывод сообщения об этом в логах и возврат исходного текста
            print(colored("Not translated phrase: {}".format(text), "red"))
            return text


class OwnerPerson:

    name = ""
    home_city = ""
    native_language = ""
    target_language = ""


class VoiceAssistant:

    name = ""
    sex = ""
    speech_language = ""
    recognition_language = ""


def setup_assistant_voice():

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


def record_and_recognize_audio():

    with microphone:
        recognized_data = ""

        # запоминание шумов окружения для последующей очистки звука от них
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

        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language=assistant.recognition_language).lower()

        except speech_recognition.UnknownValueError:
            play_voice_assistant_speech("What did you say again?")

        return recognized_data


def play_voice_assistant_speech(text_to_speech):

    ttsEngine.say(str(text_to_speech))
    ttsEngine.runAndWait()


def play_failure_phrase():

    failure_phrases = [
        translator.get("Can you repeat, please?"),
        translator.get("What did you say again?")
    ]
    play_voice_assistant_speech(failure_phrases[random.randint(0, len(failure_phrases) - 1)])


def play_greetings():

    greetings = [
        translator.get("Hello, {}! How can I help you today?").format(person.name),
        translator.get("Good day to you {}! How can I help you today?").format(person.name)
    ]
    play_voice_assistant_speech(greetings[random.randint(0, len(greetings) - 1)])


def play_farewell_and_quit():

    farewells = [
        translator.get("Goodbye, {}! Have a nice day!").format(person.name),
        translator.get("See you soon, {}!").format(person.name)
    ]
    play_voice_assistant_speech(farewells[random.randint(0, len(farewells) - 1)])
    ttsEngine.stop()
    quit()


def search_for_term_on_google():

    #if not args[0]:return
    search_term = " "#.join(args[0])
    url = "https://google.com/search?q=" + search_term
    webbrowser.get().open(url)

    search_results = []
    try:
        for _ in search(search_term,
                        tld="com",
                        lang=assistant.speech_language,
                        num=1,
                        start=0,
                        stop=1,
                        pause=1.0,
                        ):
            search_results.append(_)
            webbrowser.get().open(_)

    except (ValueError, Exception):
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()
        return

    print(search_results)
    play_voice_assistant_speech(translator.get("Here is what I found for {} on google").format(search_term))


def get_translation(*args: tuple):

    #if not args[0]:return

    search_term = " "#.join(args[0])
    google_translator = googletrans.Translator()

    old_assistant_language = assistant.speech_language
    try:
        if assistant.speech_language != person.native_language:
            translation_result = google_translator.translate(search_term,  # что перевести
                                                             src=person.target_language,  # с какого языка
                                                             dest=person.native_language)  # на какой язык

            play_voice_assistant_speech("The translation for {} in Russian is".format(search_term))

            assistant.speech_language = person.native_language
            setup_assistant_voice()

        else:
            translation_result = google_translator.translate(search_term,  # что перевести
                                                             src=person.native_language,  # с какого языка
                                                             dest=person.target_language)  # на какой язык
            play_voice_assistant_speech("По-английски {} будет как".format(search_term))

            assistant.speech_language = person.target_language
            setup_assistant_voice()

        play_voice_assistant_speech(translation_result.text)

    except (ValueError, Exception):
        play_voice_assistant_speech(translator.get("Seems like we have a trouble. See logs for more information"))
        traceback.print_exc()

    finally:
        assistant.speech_language = old_assistant_language
        setup_assistant_voice()


def change_language(*args: tuple):

    assistant.speech_language = "ru" if assistant.speech_language == "en" else "en"
    setup_assistant_voice()
    print(colored("Language switched to " + assistant.speech_language, "cyan"))


def toss_coin(*args: tuple):

    flips_count, heads, tails = 3, 0, 0

    for flip in range(flips_count):
        if random.randint(0, 1) == 0:
            heads += 1

    tails = flips_count - heads
    winner = "Tails" if tails > heads else "Heads"
    play_voice_assistant_speech(translator.get(winner) + " " + translator.get("won"))


config = {
    "intents": {
        "greeting": {
            "examples": ["привет", "здравствуй", "добрый день",
                         "hello", "good morning"],
            "responses": play_greetings
        },
        "farewell": {
            "examples": ["пока", "до свидания", "увидимся", "до встречи",
                         "goodbye", "bye", "see you soon"],
            "responses": play_farewell_and_quit
        },
        "google_search": {
            "examples": ["найди в гугл",
                         "search on google", "google", "find on google"],
            "responses": search_for_term_on_google
        },
        "translation": {
            "examples": ["выполни перевод", "переведи", "найди перевод",
                         "translate", "find translation", "interpretation",
                         "translation", "перевод", "перевести", "переведи"],
            "responses": get_translation
        },
        "language": {
            "examples": ["смени язык", "поменяй язык",
                         "change speech language", "language"],
            "responses": change_language
        },
        "toss_coin": {
            "examples": ["подбрось монетку", "подкинь монетку",
                         "toss coin", "coin", "flip a coin"],
            "responses": toss_coin
        }
    },

    "failure_phrases": play_failure_phrase
}


def prepare_corpus():

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

    best_intent = classifier.predict(vectorizer.transform([request]))[0]

    index_of_best_intent = list(classifier_probability.classes_).index(best_intent)
    probabilities = classifier_probability.predict_proba(vectorizer.transform([request]))[0]

    best_intent_probability = probabilities[index_of_best_intent]

    # при добавлении новых намерений стоит уменьшать этот показатель
    print(best_intent_probability)
    if best_intent_probability > 0.157:
        return best_intent


def make_preparations():

    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, \
        classifier

    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    ttsEngine = pyttsx3.init()

    person = OwnerPerson()
    person.name = "Лена"
    person.home_city = "Белгород"
    person.native_language = "ru"
    person.target_language = "en"

    assistant = VoiceAssistant()
    assistant.name = "Ruble"
    assistant.sex = "male"
    assistant.speech_language = "en"

    setup_assistant_voice()

    translator = Translation()

    load_dotenv()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()


if __name__ == "__main__":
    make_preparations()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()

    while True:
        # старт записи речи с последующим выводом распознанной речи
        # и удалением записанного в микрофон аудио
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        # отделение команд от дополнительной информации (аргументов)
        if voice_input:
            voice_input_parts = voice_input.split(" ")

            # если было сказано одно слово - выполняем команду сразу
            # без дополнительных аргументов
            if len(voice_input_parts) == 1:
                intent = get_intent(voice_input)
                if intent:
                    config["intents"][intent]["responses"]()
                else:
                    config["failure_phrases"]()

            # в случае длинной фразы - выполняется поиск ключевой фразы
            # и аргументов через каждое слово,
            # пока не будет найдено совпадение
            if len(voice_input_parts) > 1:
                for guess in range(len(voice_input_parts)):
                    intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    if intent:
                        command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        config["intents"][intent]["responses"](*command_options)
                        break
                    if not intent and guess == len(voice_input_parts) - 1:
                        config["failure_phrases"]()
