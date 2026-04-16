import nltk
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Download the NLTK data needed for tokenization and lemmatization.
for package in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(package, quiet=True)


class NLPChatbot:
    """Simple intent-based chatbot using TF-IDF and cosine similarity."""

    def __init__(self, intents_data):
        """Store the training data and prepare the vectorizer."""

        # Validate intent data structure
        for intent in intents_data:
            if not all(key in intent for key in ['patterns', 'tag', 'responses']):
                raise ValueError(f"Invalid intent structure. Each intent must have 'patterns', 'tag', and 'responses' keys: {intent}")
            if not intent['patterns'] or not intent['responses']:
                raise ValueError(f"Intent '{intent.get('tag', 'unknown')}' has empty patterns or responses.")

        self.intents = intents_data
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        # Build a translation table to remove punctuation during preprocessing.
        self.remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

        # The custom tokenizer normalizes text before vectorization.
        # preprocessor=None and lowercase=False prevent double lowercasing.
        # stop_words is kept as None so short identity queries like
        # "who are you" are not stripped down to empty tokens.
        self.vectorizer = TfidfVectorizer(tokenizer=self.normalize_text, preprocessor=None, lowercase=False, stop_words=None)

        # These lists hold the training phrases and their matching intent tags.
        self.corpus_patterns = []
        self.pattern_tags = []

        # Stores the last user message if future features need conversation memory.
        self.last_question = None

        self._train_model()

    def normalize_text(self, text):
        """Lowercase, remove punctuation, tokenize, and lemmatize text."""

        text = text.lower().translate(self.remove_punct_dict)

        tokens = nltk.word_tokenize(text)

        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def _train_model(self):
        """Convert intent patterns into a TF-IDF matrix for similarity search."""

        for intent in self.intents:

            for pattern in intent['patterns']:

                self.corpus_patterns.append(pattern)

                self.pattern_tags.append(intent['tag'])

        # Ensure we have patterns to train on
        if not self.corpus_patterns:
            raise ValueError("No training patterns found. Ensure intents_data contains at least one pattern.")

        # Learn vocabulary and vector representations from all known patterns.
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_patterns)

    def get_response(self, user_input):
        """Return the most relevant response for the given user input."""

        # Transform the user's message into the same vector space as the training data.
        user_vector = self.vectorizer.transform([user_input])

        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix)

        best_match_index = similarity_scores.argmax()

        highest_score = similarity_scores[0][best_match_index]

        # If the best match is too weak, fall back to a generic help message.
        # A threshold of 0.25 balances precision and recall for short queries.
        if highest_score < 0.25:

            return "I’m not sure I understand. Try asking about Python, NLP, AI, or your internship."

        matched_tag = self.pattern_tags[best_match_index]

        # Find the intent with the matched tag and return one random reply.
        for intent in self.intents:

            if intent['tag'] == matched_tag:

                return random.choice(intent['responses'])

        return "Something went wrong while generating the response."

    def start_chat(self):
        """Run the chatbot in a terminal loop until the user exits."""

        print("=" * 60)
        print("         NLP AI CHATBOT")
        print("=" * 60)

        print("Bot: Hello! I am Nexus, your NLP assistant.")
        print("Bot: Ask me about Python, NLP, AI, or your internship.")
        print("Bot: Type 'quit', 'exit', or 'bye' to end the chat.\n")

        while True:

            try:

                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # End the chat when the user types a supported exit keyword.
                if user_input.lower() in ['quit', 'exit', 'bye']:

                    print("Bot:", self.get_response("bye"))

                    break

                # Generate and display the chatbot's response.
                response = self.get_response(user_input)

                print("Bot:", response)

                self.last_question = user_input

            except KeyboardInterrupt:

                print("\nBot: Chat session ended safely. Goodbye!")

                break

            except Exception as e:

                print("\nBot: Unexpected error:", e)


if __name__ == "__main__":

    # Intent database: each intent contains example inputs and possible replies.
    INTENTS_DB = [

        {
            "tag": "greeting",
            "patterns": ["hello", "hi", "hey there", "good morning", "greetings", "what's up"],
            "responses": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to know?",
                "Greetings! Ready to assist."
            ]
        },

        {
            "tag": "goodbye",
            "patterns": ["bye", "see you later", "goodbye", "exit", "quit", "I am done"],
            "responses": [
                "Goodbye! Have a great day.",
                "See you later! Good luck with your tasks.",
                "Bye! Shutting down."
            ]
        },

        {
            "tag": "identity",
            "patterns": ["who are you", "what is your name", "are you human", "what are you"],
            "responses": [
                "I am Nexus, an NLP chatbot built using Python and NLTK.",
                "I am an AI assistant created for your NLP project."
            ]
        },

        {
            "tag": "codtech_internship",
            "patterns": ["tell me about the internship", "what are my tasks", "how many tasks do I have", "codtech info"],
            "responses": ["For the CodTech internship, you must attempt four out of four tasks and submit them before the deadline. A completion certificate will be issued on your internship end date!"]
        },

        {
            "tag": "nlp_explanation",
            "patterns": [
                "what is nlp",
                "explain natural language processing",
                "how does nlp work"
            ],
            "responses": [
                "Natural Language Processing is a branch of AI that enables computers to understand, interpret, and process human language."
            ]
        },

        {
            "tag": "ai_explanation",
            "patterns": [
                "what is ai",
                "define ai",
                "ai meaning",
                "what is artificial intelligence",
                "tell me about ai"
            ],
            "responses": [
                "Artificial Intelligence is the field of building systems that can perform tasks requiring human-like intelligence, such as learning, reasoning, and decision-making."
            ]
        },

        {
            "tag": "python",
            "patterns": [
                "what is python",
                "tell me about python",
                "why use python"
            ],
            "responses": [
                "Python is a powerful high-level programming language widely used in AI, data science, and web development."
            ]
        },

        {
            "tag": "machine_learning",
            "patterns": [
                "what is machine learning",
                "define machine learning",
                "ml meaning"
            ],
            "responses": [
                "Machine Learning is a subset of AI that allows systems to learn patterns from data and improve automatically."
            ]
        }
    ]

    bot = NLPChatbot(INTENTS_DB)

    bot.start_chat()