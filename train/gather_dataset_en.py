import pandas as pd


# Dataset sources
csv_sources = [
    "https://huggingface.co/datasets/Jarbas/ovos_intent_examples/resolve/main/dataset.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_templates/resolve/main/music_templates.csv",
    "https://raw.githubusercontent.com/OpenVoiceOS/lang-support-tracker/refs/heads/dev/skills/intents_en.csv",
]

BLACKLIST_SKILLS = [
    "ovos-skill-local-media.openvoiceos",
    "ovos-skill-spotify.openvoiceos"
]
SKILL_REPLACEMENTS = {

}
BLACKLIST_INTENTS = [

]
INTENT_REPLACEMENTS = {  # merge similar intents
    "is_rain": "do-i-need-an-umbrella.intent",
    "howto.intent": "wikihow.intent",
    "howareyou.intent": "greetings.intent",
    "handle_show_time": "handle_query_time"
}

def normalize(text):
    return str(text).lower().replace(",", "").split("/")[-1].replace("  ", " ").strip().strip('"')

def normalize_domain(text):
    n = str(text).strip().strip('"')
    for k, v in SKILL_REPLACEMENTS.items():
        n = n.replace(k, v)
    return n.replace("skill-ovos", "ovos-skill")

def normalize_intent(text):
    n = str(text).strip().strip('"')
    for k, v in INTENT_REPLACEMENTS.items():
        n = n.replace(k, v)
    return n


def load_and_format_csv(url):
    try:
        df = pd.read_csv(url)

        if "github" not in url:
            lang = "en"
        else:
            lang = url.split("_")[-1].split(".csv")[0]

        if "music_templates" in url:
            df["domain"] = "ocp"
            df["intent"] = "play"
            df = df.rename(columns={"template": "sentence"})
        else:
            if "utterance" in df.columns:
                df = df.rename(columns={"utterance": "sentence"})

        df = df[["domain", "intent", "sentence"]]

        # Normalize all columns
        df["domain"] = df["domain"].apply(normalize_domain)
        df["intent"] = df["intent"].apply(normalize_intent)
        df["sentence"] = df["sentence"].apply(normalize)

        # Drop blacklisted domains
        df = df[~df["domain"].isin(BLACKLIST_SKILLS)]
        df = df[~df["intent"].isin(BLACKLIST_INTENTS)]

        df["label"] = df["domain"] + ":" + df["intent"]
        df["lang"] = lang
        return df[["lang", "label", "sentence"]]
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return pd.DataFrame(columns=["lang", "label", "sentence"])


# Load and merge all datasets
frames = [load_and_format_csv(url) for url in csv_sources]
merged_df = pd.concat(frames, ignore_index=True)

# Deduplicate
merged_df.drop_duplicates(inplace=True)

# Save to CSV
merged_df.to_csv("merged_intents_dataset_en.csv", index=False)

print(f"Merged dataset created with {len(merged_df)} unique examples.")
