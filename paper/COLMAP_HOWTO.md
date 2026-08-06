# COLMAP-Vergleich nachholen (Windows, dieser PC, dieses Repo)

Der Vergleich mit COLMAP ist der einzige inhaltliche Punkt aus den Gutachten, der im Paper
noch fehlt. Er fehlt nur deshalb, weil auf diesem Rechner kein COLMAP installiert ist.
Diese Anleitung führt in sechs Schritten dorthin. Rechenzeit insgesamt etwa eine bis zwei
Stunden, davon ist der größte Teil Wartezeit.

Alle Befehle werden in **PowerShell** ausgeführt, aus dem Repo-Verzeichnis heraus:

```powershell
cd "C:\Users\Rudolf\ZEUG\Studium\Master IIW\1. Semester\3d Datenvisualosierung\sfm_simple\sfm-pub"
```

---

## Schritt 1: COLMAP herunterladen

COLMAP muss nicht installiert werden, es ist ein entpacktes Verzeichnis.

1. Auf <https://github.com/colmap/colmap/releases> die neueste Version öffnen.
2. Die Datei **`COLMAP-<version>-windows-no-cuda.zip`** herunterladen.
   Die CUDA-Variante bringt hier nichts: Sie wird nur für die dichte Rekonstruktion
   (Patch-Match-Stereo) gebraucht, und die soll für den Vergleich gar nicht laufen. Die
   Sparse-Rekonstruktion, um die es geht, rechnet auf der CPU.
3. Das ZIP nach `C:\tools\colmap` entpacken, so dass danach die Datei
   `C:\tools\colmap\bin\colmap.exe` existiert.

Prüfen, ob es startet:

```powershell
& "C:\tools\colmap\bin\colmap.exe" help
```

Es muss eine Liste von Unterbefehlen erscheinen (`feature_extractor`, `mapper`, …).
Kommt stattdessen eine Fehlermeldung über fehlende DLLs, stattdessen
`C:\tools\colmap\COLMAP.bat` verwenden; überall unten, wo `--colmap-bin` steht, dann
diesen Pfad einsetzen.

> Der Pfad wird gleich per `--colmap-bin` direkt übergeben. An der PATH-Variable von
> Windows muss nichts geändert werden.

## Schritt 2: Einen sauberen Bilderordner anlegen

Der Originalordner des Buddha-Datensatzes enthält neben den 67 PNGs auch `_P.txt`- und
`_seeds.bin`-Dateien. COLMAP versucht, jede Datei im Ordner als Bild zu lesen, deshalb
bekommt es einen eigenen Ordner mit ausschließlich Bildern:

```powershell
New-Item -ItemType Directory -Force C:\sfm_daten\buddha67 | Out-Null
Copy-Item "C:\Users\Rudolf\Downloads\dataset_buddha-master\dataset_buddha-master\buddha\*._c.png" C:\sfm_daten\buddha67
(Get-ChildItem C:\sfm_daten\buddha67).Count      # muss 67 ergeben
```

## Schritt 3: COLMAP über das Repo laufen lassen

Wichtig ist, dass COLMAP über denselben Einstiegspunkt läuft wie die eigene Pipeline. Nur
so sind Eingabedaten und Messweise identisch.

```powershell
.\venv\Scripts\python.exe run_sfm.py `
  --image_dir C:\sfm_daten\buddha67 `
  --output paper_out\colmap.ply `
  --backend colmap `
  --colmap-bin "C:\tools\colmap\bin\colmap.exe" `
  --colmap-keep-workspace `
  --verbose
```

Was dabei passiert: Das Repo ruft nacheinander `feature_extractor`, `exhaustive_matcher`
und `mapper` auf, liest anschließend das Ergebnis aus dem COLMAP-Modell und schreibt es als
`paper_out\colmap.ply` im selben Format wie die eigene Pipeline.

`--colmap-keep-workspace` sorgt dafür, dass der Zwischenordner
`paper_out\colmap_workspace` erhalten bleibt. Ohne diesen Schalter wird er am Ende
gelöscht, und die Brennweitenprüfung weiter unten wäre nicht mehr möglich. Im Log steht die
Zeile `[COLMAP] Best model : …`; dieser Pfad wird dort gebraucht.

Nicht `--dense` mitgeben. Die dichte Rekonstruktion von COLMAP setzt eine CUDA-Grafikkarte
voraus und wird für den Vergleich nicht benötigt.

Rechenzeit auf diesem Rechner: grob 30 bis 60 Minuten, da alles auf der CPU läuft. Das
Fenster nicht schließen. Solange Zeilen mit `[COLMAP]` erscheinen, arbeitet es.

## Schritt 4: Die eigene Pipeline auf genau denselben Bildern laufen lassen

Damit der Vergleich fair ist, muss die eigene Wolke aus demselben Ordner stammen. Die
Parameter entsprechen dem Referenzlauf aus dem Paper:

```powershell
.\venv\Scripts\python.exe run_sfm.py `
  --image_dir C:\sfm_daten\buddha67 `
  --output paper_out\own.ply `
  --n_features 12000 --ratio 0.7 `
  --verbose
```

Rechenzeit etwa 25 Minuten.

> Die vorhandene Datei `buddha_python_dense.ply` stammt aus demselben Datensatz und ließe
> sich theoretisch wiederverwenden. Für die Laufzeitspalte der Tabelle ist ein frischer
> Lauf trotzdem besser, weil beide Zeiten dann auf demselben Rechnerzustand gemessen sind.

## Schritt 5: Die Vergleichsabbildung erzeugen

```powershell
.\venv\Scripts\python.exe paper\scripts\compare_ply.py `
  paper_out\own.ply paper_out\colmap.ply `
  -o paper\figures\fig14_vergleich_colmap.png `
  --labels "Eigene Pipeline" "COLMAP" --elev 20 --azim -60 --dpi 200
```

Das Skript rendert beide Wolken aus derselben Blickrichtung und mit derselben Skalierung
nebeneinander. Sitzt der Blickwinkel ungünstig, `--elev` und `--azim` variieren, zum
Beispiel `--elev 10 --azim -120`.

Die Endung **`.png`** ist wichtig: Der PDF-Export bindet Bilder über HTML ein und kann kein
PDF als Bild darstellen.

## Schritt 6: Die Kennzahlentabelle erzeugen

```powershell
.\venv\Scripts\python.exe paper\scripts\benchmark.py `
  --image_dir C:\sfm_daten\buddha67 `
  --backends python colmap `
  --out-dir paper_out\benchmark `
  -- --n_features 12000 --ratio 0.7 --colmap-bin "C:\tools\colmap\bin\colmap.exe"
```

Alles nach dem einzelnen `--` wird an beide Läufe weitergereicht. Das Skript startet beide
Backends noch einmal, misst die Laufzeit und schreibt `paper_out\benchmark\table1.md`,
eine fertige Markdown-Tabelle mit Laufzeit, registrierten Kameras, Punktzahl und
Reprojektionsfehler.

> Dieser Schritt wiederholt die Läufe aus Schritt 3 und 4 und dauert entsprechend noch
> einmal so lange. Wer die Zeit sparen will, überspringt Schritt 6 und trägt die Zahlen aus
> den Logausgaben von Schritt 3 und 4 von Hand in die Tabelle ein.

---

## Ergebnisse ins Paper übernehmen

1. **Abbildung.** In `paper/paper_t.md` in Abschnitt 4.2 nach Fig. 11 einfügen:

   ```markdown
   ![Vergleich mit COLMAP](figures/fig14_vergleich_colmap.png)

   **Fig. 14:** Dieselbe Szene, links die eigene Pipeline, rechts COLMAP, in identischer
   Ansicht und Skalierung. [Beobachtung eintragen: Dichte, Rauschen, Vollständigkeit.]
   ```

2. **Tabelle.** Die Spalten aus `paper_out\benchmark\table1.md` in Tab. 3 in
   Abschnitt 4.3 als weitere Spalte „COLMAP" ergänzen. Die RMSE-Zelle von COLMAP bleibt
   leer, weil COLMAP diesen Wert nicht protokolliert.

3. **Text.** Zwei Stellen im Paper sind darauf vorbereitet und müssen angepasst werden:
   der letzte Absatz von Abschnitt 3.6 („kein COLMAP-Binary installiert") und der
   Abschnitt „Offene Punkte" in Anhang B.

4. **Neu bauen.**

   ```powershell
   .\venv\Scripts\python.exe paper\scripts\build_paper.py --pdf
   ```

## Worauf beim Vergleich zu achten ist

- **Die Brennweite ist der interessanteste Punkt.** Die eigene Pipeline rät bei fehlendem
  EXIF `focal = max(W, H)` und liegt damit 47 % daneben (Abschnitt 4.3). COLMAP setzt
  einen ähnlichen Startwert, verfeinert ihn aber im eigenen Bundle Adjustment. Ein Blick
  in die COLMAP-Kameradatei zeigt, wie nah es an die Ground Truth von 1.860,9 px kommt:

  ```powershell
  & "C:\tools\colmap\bin\colmap.exe" model_converter `
    --input_path paper_out\colmap_workspace\sparse\0 `
    --output_path paper_out\colmap_txt --output_type TXT
  Get-Content paper_out\colmap_txt\cameras.txt | Select-Object -First 5
  ```

  Falls `sparse\0` nicht existiert, den Pfad aus der Logzeile
  `[COLMAP] Best model : …` von Schritt 3 einsetzen. Der dritte Zahlenwert in der
  Kamerazeile ist die geschätzte Brennweite in Pixeln. Fällt sie deutlich näher an
  1.860,9 als 2.736,0, ist das der direkte Beleg für die Aussage in Abschnitt 5, dass eine
  Brennweitensuche zur Initialisierung den entscheidenden Unterschied macht.

- **Die Laufzeiten sind nur auf demselben Rechner vergleichbar.** Beide Läufe müssen also
  auf diesem PC stattfinden, und während der Messung sollte nichts anderes Größeres laufen.

- **COLMAP wird vermutlich mehr Punkte und weniger Rauschen liefern.** Das ist das
  erwartete Ergebnis und kein Problem für das Paper. Interessant für die Diskussion ist
  die Frage, woran es liegt, und dafür sind die Kamerazahl, die Punktzahl und die
  geschätzte Brennweite die aussagekräftigsten Größen.

## Wenn etwas schiefgeht

| Meldung | Ursache und Abhilfe |
|---|---|
| `COLMAP executable not found` | Pfad in `--colmap-bin` stimmt nicht. Mit `Test-Path "C:\tools\colmap\bin\colmap.exe"` prüfen. |
| COLMAP bricht beim Einlesen ab | Es liegen Nicht-Bilddateien im Ordner. Schritt 2 wiederholen. |
| `mapper` findet nur wenige Kameras | Meist zu wenige verifizierte Paare. Im Log die Zahl der Matches prüfen; notfalls den Lauf mit dem vollen Datensatz statt einer Teilmenge wiederholen. |
| Sehr langsam | Erwartungsgemäß, weil SIFT und Matching ohne CUDA auf der CPU laufen. Laufen lassen und nebenbei etwas anderes tun. |
| Speicher läuft voll | Andere Programme schließen. Der Sparse-Lauf braucht deutlich unter 8 GB. |
