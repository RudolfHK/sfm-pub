# COLMAP-Vergleich (Windows, dieser PC, dieses Repo)

Der Vergleich mit COLMAP war der letzte inhaltliche Punkt aus den Gutachten, der im Paper
gefehlt hat. Er ist inzwischen durchgeführt; die Ergebnisse stehen in Tab. 3 und Fig. 12
von `paper_t.md`. Diese Anleitung beschreibt den Weg dorthin, damit er nachvollziehbar und
wiederholbar bleibt.

Gesamtdauer auf diesem Rechner: **knapp 5 Minuten** für COLMAP, dazu die Zeit für den
eigenen Vergleichslauf.

Alle Befehle laufen in **PowerShell** aus dem Repo-Verzeichnis:

```powershell
cd "C:\Users\Rudolf\ZEUG\Studium\Master IIW\1. Semester\3d Datenvisualosierung\sfm_simple\sfm-pub"
```

---

## Schritt 1: COLMAP bereitstellen

COLMAP muss nicht installiert werden, es ist ein entpacktes Verzeichnis.

1. Auf <https://github.com/colmap/colmap/releases> die neueste Version öffnen.
2. **`COLMAP-<version>-windows-no-cuda.zip`** herunterladen. Die CUDA-Variante bringt hier
   nichts, denn CUDA wird nur für die dichte Rekonstruktion gebraucht, und die läuft für
   diesen Vergleich gar nicht.
3. Nach `C:\tools\colmap` entpacken, so dass `C:\tools\colmap\bin\colmap.exe` existiert.

Prüfen:

```powershell
& "C:\tools\colmap\bin\colmap.exe" help
```

Es erscheint eine Liste von Unterbefehlen. Bei Fehlermeldungen über fehlende DLLs
stattdessen `C:\tools\colmap\COLMAP.bat` verwenden und diesen Pfad überall dort einsetzen,
wo unten `--colmap-bin` steht. An der PATH-Variable von Windows muss nichts geändert
werden.

## Schritt 2: Sauberen Bilderordner anlegen

Der Originalordner des Buddha-Datensatzes enthält neben den 67 PNGs auch `_P.txt`- und
`_seeds.bin`-Dateien. COLMAP versucht, jede Datei als Bild zu lesen, deshalb bekommt es
einen eigenen Ordner:

```powershell
New-Item -ItemType Directory -Force C:\sfm_daten\buddha67 | Out-Null
Copy-Item "C:\Users\Rudolf\Downloads\dataset_buddha-master\dataset_buddha-master\buddha\*._c.png" C:\sfm_daten\buddha67
(Get-ChildItem C:\sfm_daten\buddha67).Count      # muss 67 ergeben
```

## Schritt 3: COLMAP über das Repo laufen lassen

Entscheidend ist, dass COLMAP über denselben Einstiegspunkt läuft wie die eigene Pipeline.
Nur so sind Eingabedaten und Messweise identisch.

```powershell
.\venv\Scripts\python.exe run_sfm.py `
  --image_dir C:\sfm_daten\buddha67 `
  --output paper_out\colmap.ply `
  --backend colmap `
  --colmap-bin "C:\tools\colmap\bin\colmap.exe" `
  --colmap-keep-workspace `
  --verbose
```

Das Repo ruft nacheinander `feature_extractor`, `exhaustive_matcher` und `mapper` auf und
schreibt das Ergebnis als `paper_out\colmap.ply` im selben Format wie die eigene Pipeline.

`--colmap-keep-workspace` erhält den Zwischenordner `paper_out\colmap_workspace`. Ohne
diesen Schalter wird er am Ende gelöscht, und Schritt 5 wäre nicht mehr möglich. Im Log
steht die Zeile `[COLMAP] Best model : …` mit dem Pfad des besten Modells.

Nicht `--dense` mitgeben, denn die dichte Rekonstruktion von COLMAP setzt eine
CUDA-Grafikkarte voraus und wird für den Vergleich nicht gebraucht.

**Gemessen auf diesem Rechner:** 38 s Merkmalsextraktion, 3 min 33 s Matching, 42 s Mapper,
zusammen 293 s. Ergebnis: 67 von 67 Kameras, 35.538 Punkte.

## Schritt 4: Die eigene Pipeline auf denselben Bildern

Für die Zahlen im Paper wurde die vorhandene Basiskonfiguration aus der Messreihe
verwendet (`eval_results/n67_base.ply`, ebenfalls 8.000 Merkmale, erschöpfendes Matching,
derselbe Rechner). Ein frischer Lauf geht so:

```powershell
.\venv\Scripts\python.exe run_sfm.py `
  --image_dir C:\sfm_daten\buddha67 `
  --output paper_out\own.ply `
  --n_features 8000 `
  --export-cameras paper_out\own.cameras.json `
  --verbose
```

Wichtig ist, dass beide Systeme dieselbe Merkmalszahl bekommen. Mit `--n_features 12000`
entspricht der Lauf dem Referenzlauf der Abbildungen, ist dann aber nicht mehr direkt mit
der COLMAP-Spalte vergleichbar.

## Schritt 5: COLMAP gegen die Ground Truth bewerten

Das ist der aussagekräftigste Teil des Vergleichs, denn er misst beide Systeme mit
demselben Maßstab. Zuerst das COLMAP-Modell nach TXT wandeln, dann in das Kameraformat des
Repos überführen und bewerten:

```powershell
New-Item -ItemType Directory -Force paper_out\colmap_txt | Out-Null
& "C:\tools\colmap\bin\colmap.exe" model_converter `
  --input_path paper_out\colmap_workspace\sparse\0 `
  --output_path paper_out\colmap_txt --output_type TXT

.\venv\Scripts\python.exe paper\scripts\colmap_to_cameras.py paper_out\colmap_txt `
  -o paper_out\colmap.cameras.json --n-images 67

.\venv\Scripts\python.exe eval\gt_pose_eval.py paper_out\colmap.cameras.json `
  --gt-dir "C:\Users\Rudolf\Downloads\dataset_buddha-master\dataset_buddha-master\buddha"
```

Das Zielverzeichnis muss vor `model_converter` existieren, sonst bricht der Befehl mit
`Directory does not exist` ab. Existiert `sparse\0` nicht, den Pfad aus der Logzeile
`[COLMAP] Best model : …` einsetzen.

**Gemessen auf diesem Rechner:**

```
registered      : 67/67 (100.0% of GT cameras)
focal           : est 1857.5 px vs GT 1860.9 px (-0.2%)
[lsq   ] pos median 0.02% max 0.05%  |  rot median 0.11° max 0.20°
```

Zum Vergleich die eigene Pipeline in derselben Konfiguration: Brennweite 2.736,0 px
(+47,0 %), Positionsfehler 2,06 % im Median, Rotationsfehler 6,29° im Median. Die
Brennweite ist damit der Kern des Unterschieds: COLMAP startet ebenfalls ohne
Kalibrierung, verfeinert sie aber erfolgreich, während das eigene Bundle Adjustment auf dem
Startwert stehen bleibt.

## Schritt 6: Vergleichsabbildung erzeugen

```powershell
.\venv\Scripts\python.exe paper\scripts\compare_ply.py `
  eval_results\n67_base.ply paper_out\colmap.ply `
  -o paper\figures\fig12_vergleich_colmap.png `
  --labels "Eigene Pipeline" "COLMAP" --elev 12 --azim -70 --dpi 170 `
  --normalize --point-size 1.2
```

`--normalize` ist wichtig. Monokulares SfM bestimmt die absolute Skala nicht, die beiden
Rekonstruktionen unterscheiden sich hier um den Faktor 0,64. Ohne Normierung erscheint eine
Wolke winzig neben der anderen, was nichts über die Qualität aussagt. Sitzt der Blickwinkel
ungünstig, `--elev` und `--azim` variieren.

Die Endung **`.png`** ist nötig, weil der PDF-Export Bilder über HTML einbindet und kein
PDF als Bild darstellen kann.

## Schritt 7 (optional): Kennzahlentabelle automatisch erzeugen

```powershell
.\venv\Scripts\python.exe paper\scripts\benchmark.py `
  --image_dir C:\sfm_daten\buddha67 `
  --backends python colmap `
  --out-dir paper_out\benchmark `
  -- --n_features 8000 --colmap-bin "C:\tools\colmap\bin\colmap.exe"
```

Alles nach dem einzelnen `--` geht an beide Läufe. Das Skript startet beide Backends noch
einmal und schreibt `paper_out\benchmark\table1.md`. Da die Zahlen bereits in Tab. 3 des
Papers stehen, ist dieser Schritt nur nötig, wenn die Läufe frisch wiederholt werden
sollen.

## Paper neu bauen

```powershell
.\venv\Scripts\python.exe paper\scripts\build_paper.py --pdf
```

---

## Wenn etwas schiefgeht

| Meldung | Ursache und Abhilfe |
|---|---|
| `COLMAP executable not found` | Pfad in `--colmap-bin` stimmt nicht. Mit `Test-Path "C:\tools\colmap\bin\colmap.exe"` prüfen. |
| `Failed to parse options - unrecognised option '--SiftExtraction.use_gpu'` | Behoben. COLMAP 4 hat Optionen umbenannt; das Repo erkennt die Schreibweise inzwischen selbst. Tritt es mit einer anderen Option auf, den Namen mit `& "C:\tools\colmap\bin\colmap.exe" <unterbefehl> -h` nachschlagen. |
| `Directory "..." does not exist` bei `model_converter` | Das Ausgabeverzeichnis vorher mit `New-Item -ItemType Directory -Force` anlegen. |
| COLMAP bricht beim Einlesen ab | Nicht-Bilddateien im Ordner. Schritt 2 wiederholen. |
| `mapper` findet nur wenige Kameras | Zu wenige verifizierte Paare. Im Log die Zahl der Matches prüfen. |
| Speicher läuft voll | Andere Programme schließen. Der Sparse-Lauf braucht deutlich unter 8 GB. |

## Anmerkung zur COLMAP-Version

Getestet mit **COLMAP 4.1.1 ohne CUDA**. In Version 4 heißen zwei Optionen anders als in
der 3er-Reihe: `SiftExtraction.use_gpu` wurde zu `FeatureExtraction.use_gpu` und
`SiftMatching.use_gpu` zu `FeatureMatching.use_gpu`. `sfm/colmap_backend.py` liest die
verfügbaren Optionen inzwischen aus der Hilfe des jeweiligen Unterbefehls aus und wählt die
passende Schreibweise selbst, so dass beide COLMAP-Generationen funktionieren.
