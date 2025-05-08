from model2vec.inference import StaticModelPipeline
from time import monotonic

s = monotonic()

# Load a pretrained Model2Vec model
model = StaticModelPipeline.from_pretrained("/home/miro/PycharmProjects/NLP/distilintent/m2v_intents_potion-base-2M")
e = monotonic()
print(f"took {e - s} seconds to load model")


# Predict labels
inputs = ["do you know the time"]

s = monotonic()
predicted = model.predict(inputs)
e = monotonic()
print(f"took {e - s} seconds to predict")
print(predicted)

# Get class probabilities
probs = model.predict_proba(inputs)  # shape: (n_samples, n_classes)

# Associate predictions with labels
for input_text, prob_row in zip(inputs, probs):
    print(f"\nInput: {input_text}")
    # Zip together class labels with their probabilities
    class_probs = list(zip(model.classes_, prob_row))
    # Sort by probability descending
    class_probs.sort(key=lambda x: x[1], reverse=True)
    for label, prob in class_probs[:5]:  # top 5
        print(f"  {label}: {prob:.4f}")

# output:
#
# took 0.13661377097014338 seconds to load model
# took 0.003044468001462519 seconds to predict
# ['ovos-skill-date-time.openvoiceos:what.time.is.it.intent']
#
# Input: do you know the time
#   ovos-skill-date-time.openvoiceos:what.time.is.it.intent: 0.9135
#   ovos-skill-naptime.openvoiceos:naptime.intent: 0.0205
#   ovos-skill-dictation.openvoiceos:start_dictation.intent: 0.0167
#   ovos-skill-hello-world.openvoiceos:Greetings.intent: 0.0081
#   ovos-skill-personal.OpenVoiceOS:WhoAreYou.intent: 0.0057