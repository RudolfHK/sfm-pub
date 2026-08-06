# Arbeitsauftrag: `paper_t.md` fertigstellen, Reviews einarbeiten, PDF-Export

Überarbeitete Fassung des ursprünglichen Prompts. Sie ist so formuliert, dass sie ohne
Rückfragen ausführbar ist und dass jedes Kriterium prüfbar ist.

---

## Runde 1: Bilder einbauen (abgeschlossen)

Aus `paper/paper_draft_v1.md` die Datei `paper/paper_t.md` erzeugen (Original bleibt
unverändert) und alle Platzhalter `[ABB. n]` / `[TAB. n]` durch die tatsächlich
vorhandenen Bilder und die gemessenen Zahlen ersetzen.

**Quellen:** `sfm_visualization_20260803_102114/` (Lauf B, Hauptquelle),
`sfm_visualization_20260802_153526/` (Lauf A, nur für den Reproduzierbarkeitsvergleich),
`EVALUATION_RESULTS.md` und `eval_results/*.run.json` für die Messwerte,
`paper/figures/pipeline_overview.svg` für Abb. 1.

**Regeln:**

1. Jedes Bild vor dem Einbau ansehen. Die Bildunterschrift beschreibt, was tatsächlich zu
   sehen ist. Weicht das Bild von der Erwartung des Entwurfs ab, wird der Text korrigiert
   und nicht das Bild geschönt.
2. Bilder nach `paper/figures/run_a` bzw. `run_b` kopieren, weil `sfm_visualization*/` in
   `.gitignore` steht und das Paper nicht darauf zeigen darf.
3. Alle Abbildungen einer Argumentationskette stammen aus demselben Lauf.
4. Keine erfundenen Zahlen. Werte aus verschiedenen Läufen niemals in einer Tabellenzeile
   mischen; die Quelle jeder Tabelle wird benannt.
5. Herkunft jeder Abbildung in einem Anhang dokumentieren.

## Runde 2: Reviews, Sprache, Export

### 2.1 Reviewkritik einarbeiten (`abstract/reviews.txt`)

- **Widerspruch „von Grund auf in Python" gegen SIFT, FLANN, RANSAC, Open3D.** Beide
  Gutachter fragen danach; das ist der wichtigste Punkt. Es braucht einen eigenen
  Abschnitt mit einer Tabelle, die pro Verarbeitungsstufe auflistet, welcher Aufruf aus
  einer Bibliothek stammt und welcher Code selbst geschrieben ist. Die Aufteilung ist am
  Quelltext zu verifizieren (`sfm/*.py`, `run_sfm.py`), nicht aus dem Entwurf zu
  übernehmen. Die Frage „wirklich alles Python?" ausdrücklich beantworten, mit
  Codeumfang und mit der Konsequenz für die Laufzeit.
- **Gründe für die Limitierungen bleiben unklar (Review 2).** Die Diskussion wird so
  umgebaut, dass jede Grenze ihre Ursache im Code und ihren Beleg aus einer Messung
  bekommt.
- **Ergebnisse mit konkreten Zahlen und Visualisierungen (Review 1).** Vorhandene
  Messwerte ohne Abbildung bekommen eine; fehlende Vergleichsläufe werden benannt statt
  umschrieben.
- **Der Abstract wirkte wie eine Gliederung.** Er wird zu zusammenhängendem Text mit den
  wichtigsten Ergebniszahlen.

### 2.2 Sprache

- Keine Geviertstriche (`—`) und keine Halbgeviertstriche als Satzzeichen. Stattdessen
  Komma, Doppelpunkt, Semikolon oder ein eigener Satz.
- Zusammenhängende, gegliederte Sätze statt Stichpunktlisten, wo der Inhalt es zulässt.
- Sparsam zitieren. Vier Literaturstellen genügen, jede wird im Text tatsächlich
  verwendet.
- Wiederkehrende Textbausteine und Marker im Fließtext vermeiden.

### 2.3 Formatierung und PDF

- Alle Abbildungen müssen vorhanden und verlinkt sein; ein fehlendes Bild ist ein Fehler,
  keine Warnung.
- Es braucht einen reproduzierbaren Build nach PDF ohne LaTeX-Installation. Bild und
  Bildunterschrift dürfen nicht auf verschiedene Seiten fallen, Tabellen nicht mitten im
  Umbruch reißen.

## Prüfkriterien

| Kriterium | Prüfung |
|---|---|
| Kein Geviertstrich im Text | `grep -c "—" paper/paper_t.md` ergibt 0 |
| Alle Bilder vorhanden | `build_paper.py` meldet „fehlende Bilder: 0" |
| Jede Abbildung hat eine Unterschrift | Zahl der `<figure>`-Blöcke entspricht der Zahl der Abbildungen und Tabellen |
| Zahlen belegt | jeder Wert steht so in `EVALUATION_RESULTS.md` oder in einem Bild des Laufs |
| PDF erzeugbar | `python paper/scripts/build_paper.py --pdf` schreibt `paper/paper_t.pdf` |

## Ergebnis

- `paper/paper_t.md` sowie `paper/paper_t.html` und `paper/paper_t.pdf`
- `paper/figures/` mit allen eingebundenen Bildern
- `paper/scripts/build_paper.py` (Markdown nach HTML und PDF),
  `paper/scripts/plot_scaling.py` (Skalierungsabbildung)
- Kurzbericht: was aus den Reviews übernommen wurde, welche Aussage wegen der realen
  Daten korrigiert wurde, was offen bleibt
