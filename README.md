# Extra Trees – zufällige Schwellen statt Schnittsuche – Streamlit-Demo

Viertes und letztes Stück des **Bagging-Asts** der Baumbasierten Linie der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Nachfolger von [Random Forest](../random-forest-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Extra Trees** (Geurts, Ernst, Wehenkel 2006) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/bagging-demo/random-forest-demo. Der Baumkern ist aus cart-demo/random-forest-demo übernommen und um die Zufallsschwelle erweitert (`et_tree.py`); die Bootstrap-/Mittel-Logik ist wortgleich zu random-forest-demo, nur um einen Bootstrap-Regler ergänzt.
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, scikit-learn nur in den Tests als Gegenprobe.

**Bezug zu OR:** viel weniger Rechenaufwand je Baum heißt, dass sich bei knapper Rechenzeit (etwa in einer Schleife der Tourenplanung, die viele Vorhersagen pro Sekunde braucht) ein größerer Wald oder häufigeres Neu-Trainieren ausgeht – eine Abwägung zwischen Modellgüte und Rechenbudget, nicht nur zwischen Modellen.

**Einordnung in die Reihe:** Random Forest sucht an jedem Schnitt noch die **beste** Schwelle unter den mtry Kandidatenmerkmalen. Extra Trees geht einen Schritt weiter: **eine** zufällige Schwelle je Kandidat, keine Suche. Das ist mehr Bias (schlechtere Einzelschnitte), aber weniger Varianz (unkorreliertere Bäume) – und vor allem ein Bruchteil des Rechenaufwands. Damit ist der Bagging-Ast der Linie abgeschlossen; der Boosting-Ast (AdaBoost → Gradient Boosting → XGBoost/LightGBM/CatBoost) folgt als nächstes.

```
CART → Bagging → Random Forest → Extra Trees (dieses Stück, letztes des Bagging-Asts)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale [d = 11], mtry = 3, 70 % Training / 30 % Test, Seed 7; Klassifikation "zu spät", 30 Bäume, kein Bootstrap, sofern nicht anders angegeben) |
|---|---|
| **Aufwand: geprüfte (Merkmal, Schwelle)-Paare** | ✅ **29.346** statt **795.714** bei erschöpfender Suche gleicher Konfiguration (Random Forest) – nur **3,7 %** des Aufwands, exakt nach Formel `mtry × Blätter − 1` statt `mtry × Σ(Knotengröße − 1)`. |
| Testfehler gegen erschöpfende Suche | ➖ 16,7 % gegen 16,4 % (Random Forest) – kaum schlechter, bei einem Bruchteil des Aufwands. |
| **Bias-Varianz-Zerlegung** (Random Forest gegen Extra Trees, drei Datensätze, feste Testpunkte) | ✅ Klassifikation: Varianz **−38 %** (0,0112 → 0,0069), Bias² **+19 %** (0,106 → 0,126) – der erwartete Tausch. Regression: Varianz **−19 %**, Bias² **+43 %**. Auf diesem Datensatz überwiegt der Bias: der Gesamtfehler ist am Ende höher als bei Random Forest. |
| **Testfehler gegen die Zahl der Bäume** (Mittel über drei Datensätze) | ➖ Extra Trees nähert sich mit mehr Bäumen an (18,0 % bei 10 Bäumen, 15,1 % bei 150 Bäumen), erreicht Random Forest (13,6 % bei 30 Bäumen) aber auch bei 150 Bäumen – nur 24 % von dessen Aufwand – nicht. Mehr billige Bäume schließen die Lücke nicht zuverlässig. |
| mtry = alle Merkmale | ➖ Kein Merkmalsauswahl-Effekt mehr, nur noch die Zufallsschwelle: Testfehler 17,8 % gegen 16,7 % bei erschöpfender Suche (= Bagging). |
| Mit Bootstrap | ➖ Testfehler ändert sich kaum (16,1 % gegen 16,7 % ohne Bootstrap), aber Out-of-Bag wird verfügbar (23,5 % Fehler) – zwei unabhängige Quellen der Vielfalt. |
| Kreuzprobe mit scikit-learn | ➖ Fehlerband gegen `ExtraTreesClassifier`/`ExtraTreesRegressor` (`bootstrap=False`, andere Zufallsquelle für Schwellen); Invarianten (Schwelle im Wertebereich, Mindestblattgröße) exakt geprüft; der Aufwands-Zähler stimmt exakt mit einer direkten Instrumentierung überein. |

## Was die Demo zeigt

- **Extra Trees in Aktion:** der Wald wächst Baum für Baum mit Schritt-Regler und Abspielen: links der zuletzt hinzugekommene Einzelbaum mit seiner zufälligen Wurzel-Schwelle, rechts das Wald-Mittel über zwei wählbare Merkmale.
- **Was der Wald gelernt hat:** Testfehler gegen einen Einzelbaum **und** gegen dieselbe Konfiguration mit erschöpfender Suche, Out-of-Bag-Fehler (nur mit Bootstrap) bzw. Rate-Fehler (ohne), ein Urteil, der Aufwands-Zähler, gemittelte Gini-Wichtigkeit.
- **Regler:** Aufgabe, Kriterium, Mindestblattgröße, Zahl der Bäume, Rauschmerkmale, **mtry**, **Bootstrap-Stichproben** (Kontrollkästchen), Lieferungen, falsche Etiketten, Seed.
- **Experimente auf Knopfdruck:** Testfehler und Aufwand gegen die Zahl der Bäume (Extra Trees gegen Random Forest), Bias-Varianz-Zerlegung (Extra Trees gegen Random Forest).

## Modell und Verfahren

- **Wie Random Forest:** an jedem Knoten eine zufällige Teilmenge von mtry Merkmalen.
- **Der Unterschied:** für jedes Kandidatenmerkmal eine einzige, gleichverteilt zwischen Minimum und Maximum im Knoten gezogene Schwelle statt einer Suche über alle Werte (`random_split` in `et_tree.py`). `extra=False` reproduziert exakt den erschöpfenden Kern (cart-demo/random-forest-demo).
- **Bootstrap ist ein Regler** (Standard: aus, wie bei Geurts und scikit-learns `ExtraTrees*`): aus heißt, jeder Baum sieht alle n Trainingszeilen.
- **Aufwand:** ein Knoten mit m Zeilen kostet bei erschöpfender Suche bis zu `mtry × (m − 1)` geprüfte Schwellen, Extra Trees immer genau `mtry` – unabhängig von m. Über einen ganzen Baum: `mtry × (Blätter − 1)` statt `mtry × Σ(Knotengröße − 1)`.

## Was nicht funktioniert hat / Grenzen

- **Ein einzelner Seed reichte nicht für die "mehr Bäume schließen die Lücke"-Aussage:** auf dem Standarddatensatz (Seed 7) gab es einen Zwischenstand, bei dem 60 Extra-Trees-Bäume (nur 7,5 % von Random Forests Aufwand) den RF-Testfehler bei 30 Bäumen sogar unterboten – ein Einzeldatensatz-Zufall. Erst das Mittel über drei Datensätze zeigte, dass die Lücke auch bei 150 Bäumen bestehen bleibt. Alle "mehr Bäume helfen (nicht)"-Aussagen im Text beruhen deshalb auf dem gemittelten Experiment, nie auf einem einzelnen Lauf.
- **Random Forest gewinnt auf diesem Datensatz durchgehend auf reiner Genauigkeit** – auch bei kleineren Stichproben oder mehr Etiketten-Rauschen (beides erhöht die Varianz-Empfindlichkeit) wurde Extra Trees in keiner getesteten Einstellung besser. Das ist kein Widerspruch zur Literatur (der Effekt ist bekanntermaßen datensatzabhängig), aber auf diesem Szenario zeigt sich der Varianz-Vorteil nicht als Netto-Gewinn.
- **Ohne Bootstrap gibt es keine Out-of-Bag-Schätzung:** die App zeigt dann statt eines toten Metrik-Felds den Rate-Fehler (kein Baum ohne Wald) an der gleichen Stelle – kein Regler bleibt ohne Wirkung stehen.

## Verifikation

`tests/test_algorithm.py` (18 Tests): Zufallsschwellen liegen immer im Wertebereich der Knotenzeilen, Mindestblattgröße wird eingehalten; **der Aufwands-Zähler stimmt exakt mit einer direkten Instrumentierung der Schnittsuche überein** und ist 20-fach kleiner als die erschöpfende Alternative auf dem Standarddatensatz; `extra=False` reproduziert den unveränderten Kern aus cart-demo/random-forest-demo bitgenau; Kreuzprobe gegen `ExtraTreesClassifier`/`ExtraTreesRegressor` über ein Fehlerband; ohne Bootstrap sieht jeder Baum jede Zeile und Out-of-Bag ist leer, mit Bootstrap verhält es sich wie in bagging-demo (≈ 63,2 % einzigartige Zeilen); Grenzfälle (im Knoten konstantes Kandidatenmerkmal, ein einzelner Baum).
`tests/test_claims.py` hält **jede Zahl** aus App und README fest. `tests/test_app.py` prüft die Oberfläche per AppTest (jedes Preset, Aufgabenwechsel, Bootstrap-Regler ohne toten Metrik-Platz, mtry-Grenzen bei wechselnder Merkmalszahl, Abspielen mit mehreren Bildern und schrittspezifischen Diagramm-Schlüsseln, Permalink, Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `et_tree.py` | Baumkern (aus cart-demo/random-forest-demo, um `random_split` erweitert) |
| `et_algorithm.py` | Bootstrap-Regler, Mitteln, Out-of-Bag, Aufwands-Zähler |
| `et_scenario.py` | Lieferdaten (wie cart-demo/bagging-demo/random-forest-demo) |
| `et_evaluation.py` | Analyse, Aufwands-Sweep, Bias-Varianz |
| `et_visualization.py` | Baumdiagramm, Karte, Aufwands- und Bias-Varianz-Kurven |
| `et_presets.py`, `et_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
