"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit) - wortgleich aus cart-demo
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"
CRITERIA = {"class": ("gini", "entropy"), "reg": ("variance",)}
CRITERION_LABELS = {"gini": "Gini-Unreinheit", "entropy": "Entropie", "variance": "Varianz (Fehlerquadrate)"}
DEFAULT_CRITERION = {"class": "gini", "reg": "variance"}

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3               # Rauschmerkmale (zufällig, ohne Bezug zum Ziel)
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0     # Prozent falsche Etiketten (nur Klassifikation)
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

# Extra-Trees-eigene Regler
N_TREES_MIN, N_TREES_MAX, DEFAULT_N_TREES = 1, 150, 30
LEAF_MIN, LEAF_MAX, DEFAULT_LEAF = 1, 50, 1                 # wie Random Forest: Bäume wachsen voll, kein Beschneiden
MTRY_MIN = 1                                                 # Obergrenze passt sich zur Laufzeit der Merkmalszahl an (siehe app.py)
DEFAULT_BOOTSTRAP = False                                    # Geurts' Standard (auch scikit-learns ExtraTrees*): keine Bootstrap-Stichprobe

DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr

COLORS = {"train": "#1f77b4", "test": "#d62728", "oob": "#ff7f0e", "exhaustive": "#1f77b4", "extra": "#2ca02c"}

PRESETS = {
    "🌲 Extra Trees (Standard)": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=3, bootstrap=False, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🎲 Mit Bootstrap": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=3, bootstrap=True, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "⚡ Viele billige Bäume": dict(task="class", criterion="gini", leaf=1, n_trees=60, mtry=3, bootstrap=False, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🌳 Random Forest (mtry = alle)": dict(task="class", criterion="gini", leaf=1, n_trees=DEFAULT_N_TREES, mtry=11, bootstrap=False, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "📈 Regression": dict(task="reg", criterion="variance", leaf=1, n_trees=DEFAULT_N_TREES, mtry=3, bootstrap=False, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🌲 Extra Trees (Standard)": "mtry = 3, kein Bootstrap: Testfehler 16,7 % gegen 16,4 % bei erschöpfender Suche (= Random Forest, gleiches mtry) - kaum schlechter, dabei nur 3,7 % so viele geprüfte (Merkmal, Schwelle)-Paare.",
    "🎲 Mit Bootstrap": "Dieselben Einstellungen, zusätzlich Bootstrap-Stichproben: Out-of-Bag-Fehler wird verfügbar (23,5 %), der Testfehler ändert sich kaum (16,1 %) - Bootstrap und Zufallsschwellen sind zwei unabhängige Quellen der Vielfalt.",
    "⚡ Viele billige Bäume": "60 statt 30 Bäume, trotzdem nur 7,5 % von Random Forests Aufwand (30 Bäume, erschöpfende Suche): Testfehler 16,1 % gegen 16,7 % mit 30 Extra-Trees-Bäumen und 16,4 % bei Random Forest selbst - mehr billige Bäume schließen die Lücke nicht zuverlässig (siehe Experiment).",
    "🌳 Random Forest (mtry = alle)": "mtry = 11 = alle Merkmale: kein Merkmalsauswahl-Effekt mehr, nur noch die Zufallsschwelle. Testfehler 17,8 % gegen 16,7 % bei erschöpfender Suche (mtry = alle entspricht Bagging) - der reine Effekt der Zufallsschwelle.",
    "📈 Regression": "Regression, mtry = 3, kein Bootstrap: Testfehler 13,5 min gegen 11,6 min bei erschöpfender Suche - der Abstand ist hier deutlicher als bei Klassifikation, bei nur 7,9 % des Aufwands.",
}
