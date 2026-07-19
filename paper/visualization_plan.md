# Visualisierungsplan — Paper „Photogrammetrie auf dem Laptop"

Grundlage: vollständige Inventur der `--visualize`-Suite (`sfm/visualizer.py`, 19 Figuren)
und der on-disk-Artefakte. Ziel: (A) welche **vorhandenen** Figuren wo im Paper landen,
(B) welche Figuren **fehlen oder verbessert** werden sollten — mit Pro/Contra und
Aufwand, damit die finale Auswahl bei dir liegt.

**Zentrale, für das ganze Paper relevante Erkenntnis:**
Jede der 19 Figuren wird headless auf die Platte geschrieben und kann über
`--viz-format pdf --viz-dpi 300` direkt als **vektorielles, druckfertiges PDF** erzeugt
werden. Für das Paper heißt das: Der Großteil der Abbildungen ist ohne zusätzlichen
Code „gratis" — es genügt ein einziger Pipeline-Lauf mit den richtigen Flags.

**Zweite, ebenso wichtige Erkenntnis:**
Der COLMAP-Backend-Pfad ruft den Visualizer **nicht** auf (früher `return` in
`run_sfm.py`). Die COLMAP-Punktwolke (`.ply`) wird zwar geschrieben, aber es existiert
**keine** Python-vs-COLMAP-Vergleichsabbildung. Genau die zentrale Ergebnisabbildung des
Papers (Abb. 8) muss also neu erzeugt werden.

---

## Teil A — Vorhandene Figuren → Zuordnung zu Paper-Abschnitten

Alle „FREI": kein neuer Code, nur ein `--visualize`-Lauf. Empfehlung für den Lauf:
`--visualize --viz-format pdf --viz-dpi 300 --viz-save-video --viz-samples 5`.

| Paper | Abbildung | Quelle (vorhanden) | Datei |
|---|---|---|---|
| §3.1 | Abb. 3 — SIFT-Keypoints + Dichte-Heatmap | `_render_keypoint_overlay`, `_render_density_heatmap` | `01_features/features_*.pdf`, `density_*.pdf` |
| §3.2 | Abb. 4 — FLANN-Matches (grün/rot) | `_render_match_pair` | `02_matching/matches_*.pdf` |
| §3.2 | (opt.) Match-Matrix + Konnektivitätsgraph | `_render_match_matrix`, `_render_connectivity_graph` | `02_matching/match_matrix.pdf`, `connectivity_graph.pdf` |
| §3.3 | Abb. 5 — Epipolarlinien | `_render_epipolar` | `02_matching/epipolar_*.pdf` |
| §3.4 | Abb. 6 — Wachsende Rekonstruktion | `_render_registration_step` (+ growth-GIF) | `03_reconstruction/step_*.pdf` |
| §3.4 | (opt.) Finale Kameraposen 3D | `_render_camera_poses` | `03_reconstruction/camera_poses_final.pdf` |
| §3.5 | Abb. 7 — BA-Konvergenz | `_render_ba_convergence` | `03_reconstruction/bundle_adjustment_convergence.pdf` |
| §4.2 | Abb. 9 — Punktwolke, 6 Ansichten | `_render_pointcloud_views` | `04_pointcloud/pointcloud_6views.pdf` |
| §4/§5 | (opt.) Punkt-Lifecycle, Reproj.-Fehler-Pfeile | `_render_point_lifecycle`, `_render_reprojection_errors` | `03_reconstruction/point_lifecycle.pdf`, `reprojection_errors_*.pdf` |
| §5 | Dichte-Heatmap für texturarme Regionen | `_render_density_heatmap` | `01_features/density_*.pdf` |
| Vortrag | Turntable-/Growth-GIF (nicht fürs Paper, für die 15-min-Präsentation) | `_save_turntable_video`, `_save_reconstruction_video` | `*.gif` |

→ **8 von ~11 Paper-Abbildungen sind bereits abgedeckt.** Es fehlen im Wesentlichen die
Vergleichs- und Konzeptabbildungen unten.

---

## Teil B — Lücken & Vorschläge (Entscheidung liegt bei dir)

Jeder Vorschlag mit Nutzen für *dieses* Paper, Pro/Contra, Aufwand und Empfehlung.

### V1 — Python-vs-COLMAP-Punktwolke, Seite an Seite  ·  Abb. 8 (zentral)
- **Was:** Dieselbe Szene, dieselbe Blickrichtung, links eigene Pipeline, rechts COLMAP.
- **Warum:** Das ist die Kern-Ergebnisabbildung; die gesamte These „Möglichkeiten und
  Grenzen" hängt an ihr.
- **Pro:** Höchste Aussagekraft; unmittelbar verständlich; trägt §4 und §5.
- **Contra:** Existiert nicht automatisch (COLMAP-Pfad ohne Visualizer). Beide `.ply`
  müssen identisch gerendert werden.
- **Aufwand:** **Niedrig manuell** (beide PLY in CloudCompare/MeshLab, gleiche Kamera,
  zwei Screenshots) — oder **mittel automatisiert** (kleines Skript, das beide PLY mit
  demselben Matplotlib-/Open3D-Offscreen-Renderer zeichnet).
- **Empfehlung:** **Ja, unbedingt.** Für v1 der manuelle Weg (schnell, reproduzierbar
  genug); Automatisierung nur, falls du mehrere Szenen zeigen willst.

### V2 — Quantitativer Kennzahlenvergleich  ·  Tab. 1 (+ optional Abb. 10)
- **Was:** Laufzeit, registrierte Kameras, Punktzahl, finaler RMSE — eigene Pipeline vs.
  COLMAP auf identischen Daten. Optional als Balken-/Scatter-Abbildung.
- **Warum:** Untermauert die qualitativen Aussagen mit Zahlen; erwartet in §4.3.
- **Pro:** Objektiv, kompakt, überzeugend; Daten fallen beim Lauf ohnehin an (RMSE/Zeit
  werden bereits geloggt).
- **Contra:** Erfordert diszipliniertes Benchmarking (gleiche Hardware, gleiche Bilder,
  Zeitmessung) und ein COLMAP-Setup.
- **Aufwand:** **Niedrig–mittel** (Läufe durchführen, Log-Werte in Tabelle übertragen;
  ein kleines Mess-Skript wäre komfortabel).
- **Empfehlung:** **Ja.** Tabelle sicher; die Zusatz-Abbildung nur, wenn Platz bleibt.

### V3 — Laufzeit-Skalierungskurve (BA-Engpass sichtbar machen)  ·  Abb. 10
- **Was:** Gesamtlaufzeit bzw. BA-Zeit pro Runde über der Bild-/Kameraanzahl.
- **Warum:** Belegt die zentrale Grenz-Aussage „globales BA versagt jenseits ~50 Bilder"
  visuell statt nur behauptet.
- **Pro:** Macht ein abstraktes Skalierungsargument (O(C²·P)) anschaulich; starkes
  Diskussions-Bild.
- **Contra:** Erfordert Läufe mit gestaffelter Bildzahl (z. B. 10/20/40/80); mehr Rechenzeit.
- **Aufwand:** **Mittel** (mehrere Läufe + einfacher Plot der Log-Werte).
- **Empfehlung:** **Optional, hoher Wirkungsgrad.** Wenn du eine Grenze *zeigen* statt
  nur benennen willst, ist das die stärkste einzelne Zusatzabbildung.

### V4 — Grenzfall: planare / texturarme Szene  ·  Abb. 11
- **Was:** Gegenüberstellung gelungene vs. degenerierte Rekonstruktion (planare Szene,
  bzw. Dichte-Heatmap einer texturarmen Fläche).
- **Warum:** Belegt „warum bricht die Pipeline bei planaren Szenen zusammen" konkret.
- **Pro:** Ehrlich, lehrreich, passt exakt zum Anspruch des Beitrags; die
  Dichte-Heatmap-Hälfte ist FREI.
- **Contra:** Braucht einen bewusst planaren/texturarmen Beispiel-Datensatz.
- **Aufwand:** **Niedrig** (Datensatz aufnehmen + normaler Lauf; nutzt vorhandene Figuren).
- **Empfehlung:** **Ja, günstig.** Guter Ertrag für wenig Aufwand.

### V5 — Poliertes Pipeline-Übersichtsdiagramm  ·  Abb. 1
- **Was:** Saubere Vektorgrafik der 6 Stufen (statt ASCII), als roter Faden.
- **Warum:** Orientierung für die Leserschaft; didaktischer Anker in der Einleitung.
- **Pro:** Sehr hoher didaktischer Wert; einmal gezeichnet, überall nutzbar (auch im Vortrag).
- **Contra:** Reine Handarbeit (kein Code); Zeit fürs Layout.
- **Aufwand:** **Niedrig–mittel** (z. B. draw.io/Excalidraw/TikZ).
- **Empfehlung:** **Ja.** Fast jedes SfM-Paper hat so ein Diagramm; lohnt sich klar.

### V6 — Konzeptschema Epipolargeometrie/Triangulation  ·  Abb. 2
- **Was:** Lehrbuch-Schema: zwei Kameras, Sehstrahlen, 3D-Punkt, Epipolarlinie.
- **Warum:** Erklärt das *Prinzip*; die reale Datenabbildung (Abb. 5) zeigt nur das Ergebnis.
- **Pro:** Schließt die Verständnislücke zwischen Theorie (§2.2) und Datenbild (§3.3).
- **Contra:** Handarbeit; bei knappem Platz verzichtbar, da Abb. 5 vieles trägt.
- **Aufwand:** **Niedrig** (eine schematische Vektorzeichnung).
- **Empfehlung:** **Optional.** Schön fürs Verständnis, aber am ehesten streichbar, wenn
  die 6 Seiten eng werden.

### V7 — Reprojektionsfehler-Histogramm (1-D-Residualverteilung)
- **Was:** Verteilung aller Reprojektionsfehler nach finalem BA als Histogramm.
- **Warum:** Übliche Qualitätsdarstellung; die Daten werden bereits berechnet.
- **Pro:** Standardmetrik, leicht lesbar.
- **Contra:** Inhaltlich überlappt mit Punkt-Lifecycle-Scatter (FREI) und BA-Konvergenz
  (FREI) — Grenznutzen gering.
- **Aufwand:** **Niedrig** (Daten liegen vor, ~15 Zeilen Plot-Code).
- **Empfehlung:** **Eher nein.** Nur falls du eine klassische Residualverteilung explizit
  zeigen willst; sonst redundant.

### V8 — Restyling der Diagnose-Figuren für den Druck
- **Was:** Vorhandene Figuren haben dunkle Hintergründe und dichte Annotationen
  (Diagnose-Stil). Für den Druck ggf. heller Hintergrund, größere Schrift, weniger
  Overlay, farbenblind-sichere Palette.
- **Warum:** Lesbarkeit im Schwarzweiß-/Farbdruck und Konsistenz mit dem Style Guide
  (Segoe UI, dunkelblau).
- **Pro:** Professionelleres Erscheinungsbild; bessere Lesbarkeit.
- **Contra:** Eingriff in `visualizer.py` bzw. Nachbearbeitung; die PDFs sind aber schon
  vektoriell, d. h. oft reicht leichte Anpassung.
- **Aufwand:** **Niedrig je Figur** (nachbearbeiten) bis **mittel** (globaler Stil-Schalter
  im Code).
- **Empfehlung:** **Selektiv.** Nur die 3–4 tatsächlich gedruckten Figuren anpassen, nicht
  die ganze Suite.

---

## Teil C — Empfohlenes Paket (Vorschlag zur Abstimmung)

**Minimal, hoher Ertrag (empfohlen):** V1 (Vergleich), V2 (Tabelle), V4 (Grenzfall),
V5 (Pipeline-Diagramm) — plus alle FREIEN Figuren aus Teil A. Damit ist das Paper
vollständig bebildert; Zusatzaufwand konzentriert sich auf einen COLMAP-Vergleichslauf
und zwei Handzeichnungen.

**Wenn du eine Grenze *zeigen* statt nur benennen willst:** zusätzlich V3
(Skalierungskurve) — die wirkungsvollste optionale Abbildung.

**Verzichtbar bei Platzmangel:** V6, V7; V8 nur selektiv auf gedruckte Figuren.

**Offene Voraussetzung für alle Vergleiche (V1–V3):** ein lauffähiges COLMAP-Setup und
1–3 saubere Datensätze. Der CLI-Pfad (`--backend colmap`) existiert bereits.
