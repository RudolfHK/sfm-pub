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

## Runde 3: Style Guide des Workshop-Bandes

Die Vorlage liegt unter `abstract/workshop_book_styleguide_2026/` und besteht aus
`main.tex` (XeLaTeX), `citing.bib` und den Segoe-UI-Schriftschnitten in `Fonts/`. Die
Vorlage ist verbindlich; der PDF-Export muss sie nachbilden.

**Vorgehen:** `main.tex` lesen und die Layoutparameter daraus ableiten, nicht schätzen.
Maßgeblich sind Seitenformat und Ränder, Schriftfamilie, Grund- und Überschriftengrößen,
Zeilenabstand, Satzart, Farbwerte, Ausrichtung der Überschriften, Aufbau des Titelkopfs,
Auszeichnung von Abstract und Keywords, Benennung und Platzierung der Bildunterschriften,
Tabellenstil und Zitierstil. Jede Abweichung wird begründet und im Anhang festgehalten.

**Front matter:** Titel, Autoren und Zugehörigkeit übernehmen die Angaben aus der
eingereichten Fassung `abstract/abstract_submission_v2.txt` und folgen dem Aufbau des
Autorenblocks der Vorlage. Der Abstract des Papers wird an die eingereichte Fassung
angeglichen, um die dort gegebenen Zusagen mit gemessenen Zahlen zu belegen. Zusagen, die
sich nicht einlösen lassen, werden im Paper nicht wiederholt, sondern als offener Punkt
geführt.

**Inhaltlicher Abgleich:** Aussagen der eingereichten Fassung, die den eigenen Messungen
widersprechen, werden nicht übernommen. Der Widerspruch wird stattdessen benannt und im
Bericht an die Autorenschaft aufgeführt.

**Ausgabe:** Der Export wird neu erzeugt, das PDF liegt am Ende aktualisiert vor.

## Runde 4: Kurzfassung auf sechs Seiten

Der Workshop-Band begrenzt den Beitrag auf **sechs A4-Seiten einschließlich aller
Abbildungen**. Die Langfassung `paper_t.md` bleibt unverändert erhalten; daneben entsteht
`paper/paper_6p.md` mit demselben Layout und demselben PDF-Export.

**Harte Streichungen** (nicht verhandelbar, vom Autor vorgegeben):

- Keine Anhänge. Weder ergänzende Abbildungen noch Bildherkunft noch offene Punkte.
- Abschnitte 4.1, 4.4, 4.5 und 4.6 der Langfassung entfallen ersatzlos.
- 4.2 und 4.3 werden zu einem Ergebnisabschnitt zusammengezogen.
- Von den Ergebnisabbildungen bleibt allein der Punktwolkenvergleich. Beide Wolken sollen
  darin aus derselben Blickrichtung gezeigt werden.
- Diskussion und Fazit werden gekürzt.
- Abschnitte 2 und 3 werden so weit gekürzt, wie es die Seitenzahl verlangt.

**Was trotz Kürzung erhalten bleiben muss**, weil es die Gutachten adressiert oder das
Ergebnis trägt:

1. Die Abgrenzung zwischen Bibliotheksaufruf und eigenem Code, samt Tabelle. Beide
   Gutachter haben danach gefragt.
2. Der Vergleich gegen Ground Truth und gegen COLMAP mit den gemessenen Zahlen.
3. Der Befund zur Brennweite und die daraus folgende Aussage über die
   Selbstauskunft der Pipeline.
4. Zu jeder genannten Grenze die Ursache, nicht nur das Symptom.

**Vorgehen bei der Kürzung:** Zuerst ganze Abschnitte streichen, dann Wiederholungen
zwischen Ergebnis und Diskussion auflösen, erst zuletzt Sätze verdichten. Zahlen werden
nicht gerundet oder weggelassen, um Platz zu schaffen; lieber entfällt ein ganzer Nebensatz.
Kein Inhalt wird stillschweigend abgeschwächt, nur weil der Beleg dafür gestrichen wurde.

**Abbildung mit gleicher Blickrichtung:** Zwei Rekonstruktionen derselben Szene stehen in
unterschiedlichen, willkürlichen Koordinatensystemen. Für eine gemeinsame Ansicht wird die
eine Wolke über eine Sim(3)-Anpassung auf die andere gelegt, geschätzt aus den
Kamerazentren gleichnamiger Bilder. Danach zeigt derselbe Blickwinkel auch dieselbe Seite
des Objekts.

**Prüfung:** Das erzeugte PDF hat höchstens sechs Seiten. Die Seitenzahl wird gemessen,
nicht geschätzt.

## Prüfkriterien

| Kriterium | Prüfung |
|---|---|
| Kein Geviertstrich im Text | `grep -c "—" paper/paper_t.md` ergibt 0 |
| Alle Bilder vorhanden | `build_paper.py` meldet „fehlende Bilder: 0" |
| Jede Abbildung hat eine Unterschrift | Zahl der `<figure>`-Blöcke entspricht der Zahl der Abbildungen und Tabellen |
| Zahlen belegt | jeder Wert steht so in `EVALUATION_RESULTS.md` oder in einem Bild des Laufs |
| PDF erzeugbar | `python paper/scripts/build_paper.py --pdf` schreibt `paper/paper_t.pdf` |
| Layout entspricht der Vorlage | Titelkopf, Schrift, Größen, Farbe und Ausrichtung im gerenderten PDF gegen `main.tex` geprüft |
| Bezeichner der Abbildungen | „Fig." wie in der Vorlage, kein „Abb." mehr im Text |

## Ergebnis

- `paper/paper_t.md` sowie `paper/paper_t.html` und `paper/paper_t.pdf`
- `paper/figures/` mit allen eingebundenen Bildern
- `paper/scripts/build_paper.py` (Markdown nach HTML und PDF),
  `paper/scripts/plot_scaling.py` (Skalierungsabbildung)
- Kurzbericht: was aus den Reviews übernommen wurde, welche Aussage wegen der realen
  Daten korrigiert wurde, was offen bleibt
