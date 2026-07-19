# Photogrammetrie auf dem Laptop: Möglichkeiten und Grenzen einer in Python implementierten 3D-Rekonstruktions-Pipeline

**Rudolf Hoffmann, Prof. Dr.-Ing. Frank Neumann**
HTW Berlin (Hochschule für Technik und Wirtschaft Berlin), Wilhelminenhofstr. 75a, 12459 Berlin
Rudolf.Hoffmann@Student.HTW-Berlin.de

> **Status:** Arbeitsentwurf v1 (deutsch). Fließtext ausformuliert, Abbildungs-Platzhalter
> als `[ABB. n]` bzw. `[TAB. n]` an der jeweiligen Stelle markiert. Details zu den
> Abbildungen siehe `paper/visualization_plan.md`. Zahlenwerte in eckigen Klammern
> `[…]` sind nach dem Referenzlauf einzutragen.

---

**Abstract:** Was ist möglich, wenn man eine vollständige Photogrammetrie-Pipeline von
Grund auf in Python implementiert, ohne spezialisierte Hardware, ohne kommerzielle
Bibliotheken, auf einem handelsüblichen Laptop? Dieser Frage geht der Beitrag anhand
eines studentischen Implementierungsprojekts nach. Die entwickelte Pipeline verarbeitet
eine Bildsequenz in mehreren Stufen: Detektion lokaler Merkmale (SIFT), paarweises
Matching (FLANN), robuste geometrische Verifikation, inkrementelle Kamerarekonstruktion
(PnP, Triangulation) sowie Bundle Adjustment zur Optimierung aller Kameraposen und
3D-Punkte. Optional werden Tiefenkarten (SGBM) und eine Mesh-Oberfläche
(Poisson-Rekonstruktion) erzeugt. Ein zentrales Ziel des Projekts ist es, jeden dieser
Schritte anhand von Zwischenergebnissen und Visualisierungen greifbar zu machen: von
erkannten Bildmerkmalen über übereinstimmende Punktpaare bis hin zur schrittweise
wachsenden 3D-Punktwolke. Im Beitrag werden die erzielten Ergebnisse auf realen
Datensätzen gezeigt und mit COLMAP, einem der führenden Open-Source-Werkzeuge für
3D-Rekonstruktion, verglichen. Dabei werden nicht nur die Erfolge präsentiert, sondern
auch die Grenzen ehrlich benannt: Wo versagt globales Bundle Adjustment bei mehr als
~50 Bildern? Warum bricht die Pipeline bei planar-dominanten Szenen zusammen? Und wann
ist COLMAP schlicht unersetzbar?

**Keywords:** Structure-from-Motion; Photogrammetrie; Python; COLMAP; Punktwolke;
Bundle Adjustment; Visualisierung; Studierendenprojekt

---

## 1  Einleitung

Aus einer Handvoll gewöhnlicher Fotos ein dreidimensionales Modell zu berechnen, gehört
heute zu den Standardwerkzeugen von Vermessung, Denkmalpflege, Robotik und AR/VR.
Drohnen kartieren Baustellen, Museen digitalisieren Exponate, autonome Systeme
rekonstruieren ihre Umgebung — die zugrundeliegende Technik ist in allen Fällen
*Structure-from-Motion* (SfM): die gleichzeitige Schätzung der Szenengeometrie und der
Kamerapositionen aus reinen Bilddaten.

> `[ABB. 1]` — **Pipeline-Übersichtsdiagramm** (ERSTELLT: `paper/figures/pipeline_overview.svg`):
> Bilder → Merkmale → Matching → geometrische Verifikation → inkrementelle
> Rekonstruktion → Bundle Adjustment → Punktwolke → (optional) Mesh. Als roter Faden
> für das gesamte Paper.

In der Praxis greifen die meisten Anwender zu fertigen Werkzeugen wie COLMAP, Meshroom
oder kommerziellen Lösungen wie RealityCapture. Diese liefern beeindruckende Ergebnisse,
verbergen die zugrundeliegenden Algorithmen aber als Blackbox. Der vorliegende Beitrag
verfolgt den umgekehrten Weg: Eine vollständige SfM-Pipeline wird von Grund auf in Python
implementiert — ausschließlich mit frei verfügbaren Bibliotheken (OpenCV, SciPy) und ohne
spezialisierte Hardware. Die leitende Frage lautet: *Wie weit trägt eine solche
Eigenimplementierung auf einem handelsüblichen Laptop, und wo stößt sie an ihre Grenzen?*

Der Beitrag verfolgt zwei Ziele. Erstens ein **didaktisches**: Jede Verarbeitungsstufe
wird durch Zwischenergebnisse sichtbar gemacht, sodass nachvollziehbar wird, was
zwischen Eingabebild und Punktwolke tatsächlich geschieht. Zweitens ein **evaluatives**:
Die eigenen Ergebnisse werden auf identischen Datensätzen mit COLMAP verglichen, um
sowohl die Möglichkeiten als auch die Grenzen einer reinen Python-Umsetzung ehrlich
einzuordnen.

## 2  Wissenschaftlicher Hintergrund und Motivation

### 2.1  Structure-from-Motion in Kürze

SfM geht auf die Mehrbildgeometrie der 1980er- und 1990er-Jahre zurück und wurde durch
Hartley und Zisserman [2] systematisiert. Moderne inkrementelle Systeme kulminieren in
COLMAP [3], das robuste Merkmalsverarbeitung, sorgfältige Ausreißerbehandlung und einen
effizienten C++-Bundle-Adjustment-Kern kombiniert und heute als De-facto-Referenz für
Open-Source-SfM gilt.

### 2.2  Kernalgorithmen und warum sie funktionieren

Drei geometrische Grundideen tragen die gesamte Pipeline:

- **Epipolargeometrie.** Zwei Bilder desselben Punkts sind über die *Fundamentalmatrix*
  F verbunden: Ein Punkt im ersten Bild schränkt seinen Partner im zweiten Bild auf eine
  Gerade — die *Epipolarlinie* — ein. Diese Bedingung ermöglicht es, falsche
  Korrespondenzen geometrisch zu verwerfen.
- **Triangulation.** Sind zwei Kameraposen bekannt, lässt sich ein 3D-Punkt als Schnitt
  zweier Sehstrahlen rekonstruieren. Je größer der Winkel zwischen den Strahlen
  (*Basislinie*), desto stabiler die Tiefenschätzung.
- **Bundle Adjustment (BA).** Alle Kameraposen und 3D-Punkte werden gemeinsam so
  optimiert, dass die Summe der *Reprojektionsfehler* — der Abstand zwischen gemessenem
  und zurückprojiziertem Bildpunkt — minimal wird. BA ist das numerische Herz jeder
  präzisen Rekonstruktion.

> `[ABB. 2]` — **Konzeptdiagramm Epipolargeometrie/Triangulation** (NEU): schematische
> Darstellung zweier Kameras, Sehstrahlen, 3D-Punkt und Epipolarlinie. Ergänzt die
> reale Datenabbildung in §3.3.

### 2.3  Motivation der Eigenimplementierung

Der Reiz einer Eigenimplementierung liegt in **Transparenz, Lerneffekt und
Kontrollierbarkeit**. Wer jede Stufe selbst baut, versteht nicht nur die Lehrbuchformel,
sondern auch, wo die eigentlichen Schwierigkeiten liegen: in der Parameterempfindlichkeit,
der numerischen Stabilität und der Behandlung von Randfällen. Genau diese Erkenntnisse
bleiben beim Einsatz einer Blackbox verborgen.

## 3  Verwendete Methoden

Die Pipeline gliedert sich in sechs Kernstufen (Abb. 1). Sie ist vollständig in Python auf
Basis von OpenCV und SciPy implementiert; alle Kennzahlen und Parameternamen in diesem
Abschnitt entsprechen der tatsächlichen Implementierung.

### 3.1  Merkmalsextraktion (SIFT)

Für jedes Bild werden bis zu 8.000 SIFT-Merkmale mit 128-dimensionalen Deskriptoren
detektiert. SIFT ist skalierungs-, rotations- und beleuchtungsinvariant und findet
Merkmale bevorzugt an Ecken, Kanten und texturierten Flächen. Auf Systemen mit GPU
übernimmt eine kornia-basierte Detektion; andernfalls dient OpenCV-SIFT auf der CPU als
Standard. Beide Pfade liefern denselben Ausgabe-Kontrakt.

> `[ABB. 3]` — **SIFT-Keypoints + Dichte-Heatmap** (FREI, `01_features/features_*.png`
> und `density_*.png`): Keypoint-Überlagerung auf einem Beispielbild, farbkodiert nach
> Merkmalsstärke, sowie Dichte-Heatmap zur Sichtbarmachung texturarmer Regionen.

### 3.2  Feature Matching (FLANN)

Korrespondenzen zwischen Bildpaaren werden per FLANN-Nächste-Nachbarn-Suche gefunden und
mit **Lowes Ratio-Test** (Standard 0,75) gefiltert: Nur Übereinstimmungen, deren bester
Treffer deutlich eindeutiger ist als der zweitbeste, werden behalten. Für ungeordnete
oder große Datensätze stehen zusätzlich sequenzielles Matching und ein
Vokabularbaum-Verfahren (TF-IDF über visuelle Wörter) zur Verfügung.

> `[ABB. 4]` — **Match-Visualisierung** (FREI, `02_matching/matches_*.png`): Bildpaar
> nebeneinander, Inlier grün, Ausreißer rot, mit Inlier-Quote im Titel. Optional ergänzt
> durch die Match-Matrix (`match_matrix.png`) und den Konnektivitätsgraphen
> (`connectivity_graph.png`).

### 3.3  Geometrische Verifikation (RANSAC, Fundamentalmatrix)

Jedes Match-Paar durchläuft drei aufeinanderfolgende Filter: (1) **Hartley-Normierung**
der Pixelkoordinaten zur numerischen Stabilisierung, (2) robuste Schätzung der
**Fundamentalmatrix** mittels `USAC_MAGSAC` — nur geometrisch konsistente Matches
überleben als Inlier —, und (3) Ableitung der **Essential-Matrix** und Zerlegung in die
relative Kamerapose (R, t) über die Cheiralitätsbedingung. Anschließend prüft ein
Union-Find-Verfahren die Zusammenhangskomponenten des Szenengraphen; nur die größte wird
weiterverwendet.

> `[ABB. 5]` — **Epipolarlinien** (FREI, `02_matching/epipolar_*.png`): Für einige
> Inlier-Paare werden Punkt und zugehörige Epipolarlinie farblich zusammengehörig
> gezeichnet. Liegen die Punkte exakt auf ihren Linien, ist F gut geschätzt — ein
> anschaulicher Korrektheitsnachweis.

### 3.4  Inkrementelle Rekonstruktion (Two-View-Init, PnP, Triangulation)

Die Rekonstruktion wächst kameraweise. Als **Startpaar** wird das Paar mit maximalem
`Basislinie × Inlier-Zahl` bei einem Triangulationswinkel ≥ 5° gewählt. Danach wird
iterativ jeweils das Bild mit den meisten 2D-3D-Korrespondenzen per **PnP**
(`solvePnPRansac` + LM-Verfeinerung) registriert, und neue 3D-Punkte werden mit allen
sichtbaren Kameras trianguliert. Akzeptiert werden nur Punkte mit positiver Tiefe,
hinreichendem Triangulationswinkel und Reprojektionsfehler unterhalb der Schwelle.

> `[ABB. 6]` — **Wachsende Rekonstruktion** (FREI, `03_reconstruction/step_*.png` bzw.
> `reconstruction_growth.gif`, zusätzlich `camera_poses_final.png`): Montage mehrerer
> Registrierungsschritte, die zeigt, wie Kameras und Punktwolke schrittweise entstehen.
> Kernbild für den didaktischen Anspruch des Beitrags.

### 3.5  Bundle Adjustment

Nach jeweils fünf neu registrierten Kameras (`--ba_interval`) und am Ende wird BA über
`scipy.optimize.least_squares` (Trust-Region-Reflective) mit dünnbesetzter
Jacobi-Struktur ausgeführt. Das Projektionsmodell umfasst radiale Verzeichnung
(Brown-Conrady, 2 Koeffizienten); als robuste Verlustfunktion dient ein **Huber-Loss**
mit adaptiver, aus der Streuung der Startresiduen abgeleiteter Skala. Ein
Divergenz-Schutz verwirft das BA-Ergebnis, falls sich der Fehler mehr als um den Faktor
1,5 verschlechtert.

> `[ABB. 7]` — **BA-Konvergenz** (FREI, `03_reconstruction/bundle_adjustment_convergence.png`):
> Reprojektions-RMSE vor/nach jeder BA-Runde. Zeigt anschaulich, wie der Fehler mit
> zunehmender Kamerazahl fällt und sich stabilisiert.

### 3.6  Optionale Erweiterungen und Referenzsystem

Optional erzeugt die Pipeline über StereoSGBM eine dichte Punktwolke und über
Screened-Poisson (Open3D) eine Mesh-Oberfläche. Als **Referenzsystem** dient COLMAP,
das über denselben CLI-Einstiegspunkt (`--backend colmap`) auf identischen Eingabedaten
läuft und so einen fairen Direktvergleich ermöglicht.

## 4  Ergebnisse

### 4.1  Datensätze

Ausgewertet werden [N] Datensätze mit [M]–[K] Bildern, aufgenommen mit [Kamera/Smartphone].
Ein guter SfM-Datensatz zeichnet sich durch hohe Bildüberlappung (~60–80 %), ausreichende
Textur und moderate Blickwinkeländerungen zwischen benachbarten Aufnahmen aus.

> `[ABB. 8]` — **Direktvergleich Punktwolke: eigene Pipeline vs. COLMAP** (NEU, zentrale
> Ergebnisabbildung): dieselbe Szene, links Eigenimplementierung, rechts COLMAP, in
> identischer Ansicht.

### 4.2  Qualitative Ergebnisse

Auf strukturierten Szenen mit weniger als ~50 Bildern liefert die Python-Pipeline
visuell plausible Punktwolken mit einer zu COLMAP vergleichbaren Grobstruktur; COLMAP
erzeugt jedoch dichtere und rauschärmere Wolken. [Detailbeobachtungen nach Referenzlauf.]

> `[ABB. 9]` — **Sechs-Ansichten-Punktwolke** (FREI, `04_pointcloud/pointcloud_6views.png`):
> kolorierte Punktwolke der eigenen Pipeline aus sechs orthografischen Richtungen zur
> Beurteilung von Vollständigkeit und Ausreißern.

### 4.3  Quantitativer Vergleich

> `[TAB. 1]` — **Kennzahlenvergleich** (NEU, Benchmark-Lauf erforderlich): Laufzeit,
> registrierte Kameras, 3D-Punkte und finaler Reprojektions-RMSE für eigene Pipeline
> und COLMAP auf identischen Datensätzen.

| Metrik | Python-Pipeline | COLMAP |
|---|---|---|
| Laufzeit ([N] Bilder) | [X] s | [Y] s |
| Registrierte Kameras | [K/N] | [N/N] |
| 3D-Punkte (sparse) | [A] | [B] |
| Reprojektions-RMSE | [px] | [px] |

> `[ABB. 10]` — **Laufzeit-Skalierung** (NEU, optional): Gesamtlaufzeit bzw. BA-Zeit über
> der Bildanzahl, zur Illustration des überproportionalen Anstiegs des globalen BA.

### 4.4  Visualisierung der Zwischenergebnisse

Ein Alleinstellungsmerkmal gegenüber der COLMAP-Blackbox ist die durchgängige
Visualisierbarkeit: Von der Merkmalsdichte über die Match-Matrix bis zur BA-Konvergenz
lässt sich jeder Schritt inspizieren — sowohl zu Lehrzwecken als auch zur Fehlersuche.

## 5  Diskussion

**Wo funktioniert die Python-Pipeline gut?** Bei strukturierten, gut überlappenden Szenen
mit unter ~50 Bildern und ausreichender Textur erreicht sie eine mit COLMAP vergleichbare
Rekonstruktionsqualität — ausreichend für Lehre und kleine Projekte.

**Wo versagt sie?** Drei Grenzen sind systematisch: (1) **Skalierung** — das globale BA
skaliert überproportional mit der Kamerazahl und wird jenseits von ~50 Bildern zum
Laufzeit-Engpass. (2) **Planare Szenen** — bei dominanter Ebene ist die Fundamentalmatrix
schlecht konditioniert, die Initialisierung wird instabil. (3) **Texturarme Flächen** —
SIFT findet zu wenige Merkmale, ganze Bildregionen bleiben ohne 3D-Punkte.

> `[ABB. 11]` — **Grenzfall planare/texturarme Szene** (NEU): Gegenüberstellung einer
> gelungenen und einer degenerierten Rekonstruktion, ergänzt durch die Dichte-Heatmap
> (FREI) zur Sichtbarmachung merkmalsarmer Regionen.

**Was erklärt COLMAPs Überlegenheit?** Weniger die algorithmische Grundstruktur — die ist
in beiden Systemen ähnlich — als vielmehr die **Implementierungsreife**: ein
C++-Sparse-Solver, Parallelisierung, ausgereifte Ausreißerbehandlung und jahrelanges
Tuning. Der Trade-off lautet: didaktische Transparenz und volle Kontrolle auf der einen,
Robustheit und Skalierbarkeit auf der anderen Seite.

## 6  Fazit

Eine rein Python-basierte SfM-Pipeline ist auf kleinen, für SfM geeigneten Datensätzen
durchaus wettbewerbsfähig und bietet einen erheblichen didaktischen Mehrwert durch
transparente, visualisierbare Zwischenergebnisse. Das wichtigste Learning des Projekts:
Die eigentlichen Schwierigkeiten liegen nicht in den Lehrbuchformeln, sondern in
Robustheit, Parameterempfindlichkeit und Debugging geometrischer Algorithmen. Für den
produktiven Einsatz jenseits kleiner Szenen bleibt COLMAP unersetzbar — als Lern- und
Verständniswerkzeug entfaltet die Eigenimplementierung jedoch ihren eigenen Wert.
**Ausblick:** Den größten Qualitäts- und Skalierungsgewinn verspricht lokales
(Sliding-Window-)Bundle-Adjustment, ergänzt um lernbasiertes Matching (z. B. LightGlue)
für schwierige Aufnahmesituationen.

## Literatur

[1] D. G. Lowe, „Distinctive Image Features from Scale-Invariant Keypoints", *IJCV*,
60(2), 2004.

[2] R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2. Aufl.,
Cambridge University Press, 2003.

[3] J. L. Schönberger, J.-M. Frahm, „Structure-from-Motion Revisited", *CVPR*, 2016.

[4] M. Muja, D. G. Lowe, „Fast Approximate Nearest Neighbors with Automatic Algorithm
Configuration", *VISAPP*, 2009.

[5] P. Lindenberger, P.-E. Sarlin, M. Pollefeys, „LightGlue: Local Feature Matching at
Light Speed", *ICCV*, 2023.

[6] Q.-Y. Zhou, J. Park, V. Koltun, „Open3D: A Modern Library for 3D Data Processing",
arXiv:1801.09847, 2018.
