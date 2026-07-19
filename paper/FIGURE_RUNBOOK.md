# Abbildungs-Runbook — Paket A (deutsch)

Konkrete Befehle, um jede Abbildung/Tabelle aus `paper_draft_v1.md` zu erzeugen.
Umgesetzt ist **Paket A**: V1 Vergleich, V2 Kennzahlentabelle, V4 Grenzfall,
V5 Pipeline-Diagramm — plus alle FREIEN Figuren aus der `--visualize`-Suite.

Voraussetzung einmalig:
```bash
pip install -e ".[viz,mesh]"      # matplotlib, networkx, imageio, open3d
# für V1/V2 zusätzlich ein COLMAP-Binary auf dem PATH (--backend colmap)
```

---

## Schritt 0 — Ein Referenzlauf erzeugt ALLE FREIEN Figuren

```bash
python run_sfm.py \
  --image_dir ./datensatz_gut \
  --output paper_out/own.ply \
  --match_strategy sequential --sequential_window 8 \
  --n_features 10000 \
  --visualize --viz-format pdf --viz-dpi 300 \
  --viz-samples 5 --viz-save-video \
  --viz-output paper_out/viz --verbose
```

Danach liegen die druckfertigen PDFs in `paper_out/viz_<zeitstempel>/`:

| Draft | Abbildung | Datei |
|---|---|---|
| **Abb. 3** | SIFT-Keypoints + Dichte | `01_features/features_*.pdf`, `density_*.pdf` |
| **Abb. 4** | FLANN-Matches (grün/rot) | `02_matching/matches_*.pdf` |
| (opt.) | Match-Matrix, Konnektivität | `02_matching/match_matrix.pdf`, `connectivity_graph.pdf` |
| **Abb. 5** | Epipolarlinien | `02_matching/epipolar_*.pdf` |
| **Abb. 6** | Wachsende Rekonstruktion | `03_reconstruction/step_*.pdf` + `reconstruction_growth.gif` |
| (opt.) | Finale Kameraposen 3D | `03_reconstruction/camera_poses_final.pdf` |
| **Abb. 7** | BA-Konvergenz | `03_reconstruction/bundle_adjustment_convergence.pdf` |
| **Abb. 9** | Punktwolke, 6 Ansichten | `04_pointcloud/pointcloud_6views.pdf` |
| (Vortrag) | Turntable-GIF | `04_pointcloud/pointcloud_turntable.gif` |

> Die `reconstruction_growth.gif`/Turntable-GIFs sind für den **15-min-Vortrag**
> gedacht, nicht fürs Paper. Für Abb. 6 im Paper eine Montage aus 3–4
> `step_*.pdf` (früh/mittel/spät) setzen.

---

## Abb. 1 — Pipeline-Übersicht (V5) · FERTIG

Bereits erstellt: `paper/figures/pipeline_overview.svg` (vektoriell, deutsch,
Style-Guide-Blau). Als PDF/PNG fürs Paper:
```bash
pip install cairosvg
python -c "import cairosvg; cairosvg.svg2pdf(url='paper/figures/pipeline_overview.svg', write_to='paper/figures/pipeline_overview.pdf')"
```
(oder in Inkscape/draw.io öffnen und anpassen.)

## Abb. 2 — Konzeptschema Epipolargeometrie · OFFEN (V6, optional)

Laut Plan optional. Falls gewünscht: schematische Vektorzeichnung (zwei Kameras,
Sehstrahlen, 3D-Punkt, Epipolarlinie) analog zu Abb. 1 anlegen.

## Abb. 8 — Python vs. COLMAP, Seite an Seite (V1) · TOOL FERTIG

Beide Backends auf **denselben** Bildern laufen lassen, dann rendern:
```bash
# COLMAP-Punktwolke erzeugen (Python-Wolke stammt aus Schritt 0):
python run_sfm.py --image_dir ./datensatz_gut \
  --output paper_out/colmap.ply --backend colmap

# identische Ansicht + Skala, druckfertig:
python paper/scripts/compare_ply.py \
  paper_out/own.ply paper_out/colmap.ply \
  -o paper/figures/abb8_vergleich.pdf \
  --labels "Eigene Pipeline" "COLMAP" --elev 20 --azim -60 --dpi 300
```
Blickwinkel bei Bedarf über `--elev/--azim` an die Szene anpassen.

## Tab. 1 — Kennzahlenvergleich (V2) · TOOL FERTIG

```bash
python paper/scripts/benchmark.py \
  --image_dir ./datensatz_gut \
  --backends python colmap \
  --out-dir paper_out/benchmark
# schreibt paper_out/benchmark/table1.md (direkt in den Draft übernehmbar)
```
Laufzeit-Vergleich nur aussagekräftig, wenn **beide** Backends auf derselben
Maschine laufen. COLMAP-RMSE erscheint als „n/v" (wird von COLMAP nicht geloggt).

## Abb. 11 — Grenzfall planare/texturarme Szene (V4) · HALB FREI

Einen bewusst **planaren oder texturarmen** Datensatz aufnehmen (z. B. glatte
Wand, einzelne Fassade) und Schritt 0 darauf wiederholen:
```bash
python run_sfm.py --image_dir ./datensatz_planar \
  --output paper_out/planar.ply \
  --visualize --viz-format pdf --viz-dpi 300 \
  --viz-output paper_out/viz_planar --verbose
```
- Die **Dichte-Heatmap** (`01_features/density_*.pdf`) zeigt die merkmalsarmen
  Regionen — die eine Hälfte von Abb. 11 (FREI).
- Für die andere Hälfte: gelungene vs. degenerierte Punktwolke gegenüberstellen,
  z. B. erneut mit `compare_ply.py` (gut vs. planar).

---

## Empfohlene Reihenfolge

1. Guten Datensatz wählen → **Schritt 0** (liefert Abb. 3–7, 9 + `own.ply`).
2. `--backend colmap` auf denselben Bildern → `colmap.ply`.
3. **compare_ply.py** → Abb. 8.  **benchmark.py** → Tab. 1.
4. Planaren Datensatz → **Schritt 0** darauf → Abb. 11.
5. `pipeline_overview.svg` → PDF (Abb. 1). Optional Abb. 2 zeichnen.

Danach ist das Paper vollständig bebildert; es fehlen nur noch die realen
Zahlenwerte (`[…]`) im Draft, die aus `table1.md` und den Läufen stammen.
