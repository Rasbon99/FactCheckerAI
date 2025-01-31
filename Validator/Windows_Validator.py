from transformers import pipeline

class Validator:
    def __init__(self, model_name: str):
        """
        Inizializza la classe Validator con il nome del modello.
        
        :param model_name: Il nome del modello pre-addestrato da utilizzare
        """
        self.model_name = model_name
        # Creazione della pipeline zero-shot
        self.zero_shot_classifier = pipeline("zero-shot-classification", model=self.model_name)

    def predict(self, texts, hypothesis_template: str):
        """
        Esegue la previsione sul testo dato utilizzando la classificazione zero-shot.
        
        :param texts: Una lista di testi da classificare
        :param hypothesis_template: La frase da utilizzare come ipotesi per la classificazione
        :return: I risultati della classificazione
        """
        return self.zero_shot_classifier(texts, candidate_labels=["entailment", "semi_entailment", "not_entailment"], hypothesis_template=hypothesis_template)

# Esempio d'uso
if __name__ == "__main__":
    model_name = ""
    validator = Validator(model_name)

    texts = [
        "Defending Australian Open champion Jannik Sinner has returned to the quarterfinals, navigating an illness and multiple delays. Sinner beat No. 13 seed Holger Rune 6-3, 3-6, 6- 3, 6 -2. The world No. 1 put a “strange morning behind him” in returning to the Australian Open quarterfinals. The 22-year-old is a combined 11-0 against his potential next opponents De Minaur and Michelsen.",
        "Jannik Sinner, what happened and how he is. The illness against Rune: the hands that shook and the difficulty in breathing. Very tough match for the world number 1 who beat the Dane and earned the pass to the quarterfinals of the Australian Open. Health problems between the second and third set: 'I felt dizzy, the conditions were difficult' di Sports Editorial 20 January 2025 Jannk Sinner assisted by Australian Open medical staff during match against Rune. Jannik begins to move badly on the court, It's slow and feels foggy . One point it seems to have stomach problems.",
        "Jannik Sinner wins stop-start Australian Open match with Holger Rune after illness, net problems - The Athletic Tennis. World No. 1 Sinner won in four sets, 6-3. 3-6,6-3, 6 -2. Sinner appeared to indicate to his box that he was struggling with his movement on his left side. The defending Australian Open champion winced when running down a ball from the Danish No. 13 seed on his backhand side."
    ]

    hypothesis = "The News: 'What's happening to Jannik Sinner? The Italian tennis player, number one in the world, felt ill during the match with Rune, the Dane, in the round of 16 of the Australian Open. He started to stagger in the third set, he lost his balance, he sat down during a change of court and was very pale, he was almost not breathing, he was about to vomit. Then he asked for the doctor's help, he went to the locker room for 12 minutes and came back, he was visibly better, he played a beautiful match.' is {}"

    results = validator.predict(texts, hypothesis)
    print(results)
