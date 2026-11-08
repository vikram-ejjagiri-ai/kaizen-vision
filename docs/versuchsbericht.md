# Versuchsbericht: Vergleich von Bildverarbeitungsverfahren für die Qualitätsprüfung

Kaizen Vision · Vikram Ejjagiri · Versuchslauf 2026-11-07 16:23 (90.1 min, x86_64, 2 threads)

## 1. Ziel

Für eine neue Prüfstation soll ein Bildverarbeitungsverfahren ausgewählt werden. Verglichen werden klassische Bildverarbeitung, vortrainierte und feinabgestimmte neuronale Netze sowie Anomalieerkennung: Genauigkeit, Robustheit gegenüber veränderten Aufnahmebedingungen, Bedarf an Trainingsbildern und Rechenzeit. Zusätzlich werden Leser für QR-Codes, Data-Matrix-Codes und Klartext verglichen.

## 2. Versuchsaufbau

- **Stahloberfläche (NEU-Datenbank):** 6 Fehlerarten, 1.440 Trainings- und 360 Testbilder, 200 x 200 px.
- **Gussteile (Pumpenlaufräder):** i. O. / n. i. O., 910 Trainings- und 390 Testbilder, 256 x 256 px.
- **Kennzeichnungen:** je 60 erzeugte QR-, Data-Matrix- und Klartext-Markierungen auf echter Stahltextur.
- **Rechner:** Laptop-CPU ohne GPU; vortrainiertes Netz YOLO11s-cls (ImageNet).

## 3. Versuchsdurchführung

1. Training aller Verfahren auf sauberen Bildern.
2. Prüfung auf sauberen Testbildern und unter 11 veränderten Bedingungen in je 4 Stufen (44 Bedingungen): Unter- und Überbelichtung, geringer Kontrast, ungleichmäßige Beleuchtung, Fokus- und Bewegungsunschärfe, Sensorrauschen, geringere Auflösung, Drehung, Kameraneigung, JPEG-Kompression.
3. Training mit 2 % bis 100 % der Trainingsbilder (je drei zufällige Teilmengen).
4. Messung der Rechenzeit pro Bild und der Trainingszeit.
5. Gegenmaßnahme: erneutes Training mit zufälliger Helligkeit, Unschärfe und Rauschen.
6. Anbindung: Handshake mit einer simulierten SPS über OPC UA (Normalbetrieb, Kamerafehler, Ausfall der Station) und Kamerakalibrierung für die Umrechnung von Pixeln in Millimeter.

## 4. Ergebnisse

### Stahloberfläche (6 Fehlerarten)

| Verfahren | sauber | Mittel unter veränderten Bedingungen | schlechtester Fall |
|---|---|---|---|
| klassische Merkmale + SVM | 99,7 % | 49,4 % | 16,7 % (Unschärfe (Fokus)) |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 99,4 % | 79,6 % | 40,8 % (Bewegungsunschärfe) |
| feinabgestimmtes CNN | 99,7 % | 71,8 % | 31,7 % (Unschärfe (Fokus)) |
| klassisch + SVM, mit Variation trainiert | 99,4 % | 87,2 % | 20,8 % (Sensorrauschen) |
| CNN-Merkmale, mit Variation trainiert | 98,9 % | 93,9 % | 71,7 % (Unschärfe (Fokus)) |
| feinabgestimmtes CNN, mit Variation trainiert | 100,0 % | 94,4 % | 59,2 % (JPEG-Kompression) |
Zufallsniveau: 16,7 % (6 Klassen).

**Toleranz** (höchste Stufe 0-4 innerhalb von 5 Prozentpunkten der sauberen Genauigkeit)

| | Unterbelichtung | Überbelichtung | geringer Kontrast | ungleichmäßige Beleuchtung | Unschärfe (Fokus) | Bewegungsunschärfe | Sensorrauschen | geringere Auflösung | Drehung | Kameraneigung | JPEG-Kompression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| klassische Merkmale + SVM | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 3 | 1 | 3 | 4 | 0 | 0 | 1 | 0 | 3 | 4 | 0 |
| feinabgestimmtes CNN | 1 | 1 | 1 | 4 | 0 | 0 | 0 | 0 | 3 | 4 | 0 |
| klassisch + SVM, mit Variation trainiert | 3 | 1 | 2 | 3 | 0 | 0 | 1 | 0 | 3 | 4 | 0 |
| CNN-Merkmale, mit Variation trainiert | 4 | 1 | 3 | 4 | 2 | 1 | 1 | 4 | 3 | 4 | 2 |
| feinabgestimmtes CNN, mit Variation trainiert | 3 | 2 | 3 | 4 | 2 | 2 | 2 | 4 | 3 | 4 | 2 |

**Trainingsdaten** (Genauigkeit auf dem festen Testsatz)

| Bilder pro Klasse | klassische Merkmale + SVM | vortrainierte CNN-Merkmale + linearer Klassifikator | feinabgestimmtes CNN |
|---|---|---|---|
| 5 | 86,5 % | 85,2 % | 68,1 % |
| 12 | 95,2 % | 97,2 % | 72,5 % |
| 24 | 99,4 % | 98,7 % | 90,6 % |
| 60 | 99,4 % | 98,9 % | 99,2 % |
| 120 | 99,5 % | 99,3 % | 100,0 % |
| 240 | 99,7 % | 99,4 % | 99,7 % |

| Verfahren | ms pro Bild (CPU) | Trainingszeit (s) |
|---|---|---|
| klassische Merkmale + SVM | 26,2 | 38,8 |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 7,2 | 16,5 |
| feinabgestimmtes CNN | 5,0 | 384,1 |

### Gussteil (i. O. / n. i. O.)

| Verfahren | sauber | Mittel unter veränderten Bedingungen | schlechtester Fall |
|---|---|---|---|
| klassische Merkmale + SVM | 99,2 % | 67,0 % | 54,4 % (Überbelichtung) |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 99,7 % | 83,5 % | 50,8 % (Unschärfe (Fokus)) |
| feinabgestimmtes CNN | 100,0 % | 89,9 % | 45,4 % (geringer Kontrast) |
| Anomalieerkennung | 92,8 % | 71,7 % | 60,0 % (Unterbelichtung) |
| klassisch + SVM, mit Variation trainiert | 96,9 % | 73,9 % | 37,2 % (Sensorrauschen) |
| CNN-Merkmale, mit Variation trainiert | 99,7 % | 89,4 % | 60,8 % (JPEG-Kompression) |
| feinabgestimmtes CNN, mit Variation trainiert | 99,5 % | 96,8 % | 76,9 % (JPEG-Kompression) |
Wer alles als n. i. O. meldet, erreicht hier etwa 60 % (Anteil der Fehlerteile im Testsatz).

**Toleranz** (höchste Stufe 0-4 innerhalb von 5 Prozentpunkten der sauberen Genauigkeit)

| | Unterbelichtung | Überbelichtung | geringer Kontrast | ungleichmäßige Beleuchtung | Unschärfe (Fokus) | Bewegungsunschärfe | Sensorrauschen | geringere Auflösung | Drehung | Kameraneigung | JPEG-Kompression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| klassische Merkmale + SVM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| feinabgestimmtes CNN | 1 | 2 | 1 | 4 | 1 | 1 | 2 | 2 | 4 | 4 | 1 |
| Anomalieerkennung | 2 | 0 | 1 | 3 | 1 | 0 | 0 | 1 | 1 | 2 | 0 |
| klassisch + SVM, mit Variation trainiert | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| CNN-Merkmale, mit Variation trainiert | 3 | 2 | 2 | 4 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| feinabgestimmtes CNN, mit Variation trainiert | 3 | 2 | 4 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 3 |

| Verfahren (sauber) | AUROC | Fehlerschlupf | Pseudoausschuss |
|---|---|---|---|
| klassische Merkmale + SVM | 1,000 | 0,9 % | 0,6 % |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 1,000 | 0,4 % | 0,0 % |
| feinabgestimmtes CNN | 1,000 | 0,0 % | 0,0 % |
| Anomalieerkennung | 0,981 | 7,3 % | 7,1 % |
| klassisch + SVM, mit Variation trainiert | 0,996 | 2,6 % | 3,8 % |
| CNN-Merkmale, mit Variation trainiert | 1,000 | 0,4 % | 0,0 % |
| feinabgestimmtes CNN, mit Variation trainiert | 1,000 | 0,9 % | 0,0 % |

**Trainingsdaten** (Genauigkeit auf dem festen Testsatz)

| Bilder pro Klasse | klassische Merkmale + SVM | vortrainierte CNN-Merkmale + linearer Klassifikator | feinabgestimmtes CNN |
|---|---|---|---|
| 7 | 71,5 % | 73,6 % | 71,0 % |
| 18 | 76,3 % | 80,3 % | 91,8 % |
| 36 | 83,2 % | 88,0 % | 93,1 % |
| 91 | 92,3 % | 96,8 % | 96,9 % |
| 182 | 97,4 % | 98,0 % | 99,5 % |
| 363 | 99,2 % | 99,7 % | 100,0 % |

| Verfahren | ms pro Bild (CPU) | Trainingszeit (s) |
|---|---|---|
| klassische Merkmale + SVM | 39,8 | 36,7 |
| vortrainierte CNN-Merkmale + linearer Klassifikator | 6,9 | 9,1 |
| feinabgestimmtes CNN | 6,0 | 248,5 |
| Anomalieerkennung | 14,4 | 93,4 |

### Teilekennzeichnung: Leserate

| Leser | sauber | Mittel verändert | ms |
|---|---|---|---|
| qr: OpenCV | 91,7 % | 75,4 % | 5,4 |
| qr: ZBar | 100,0 % | 77,6 % | 2,2 |
| qr: ZXing-C++ | 96,7 % | 82,4 % | 0,9 |
| datamatrix: ZXing-C++ | 93,3 % | 80,9 % | 0,9 |
| text: RapidOCR | 100,0 % | 83,1 % | 117,6 |

### Anbindung an die Steuerung (OPC UA)

Simulierte SPS (OPC-UA-Server) und Prüfstation (OPC-UA-Client) mit dem Handshake einer realen Prüfzelle: Trigger → Busy → Ergebnis, Greifposition, Winkel, Fehlercode → Done → Trigger = 0. SPS-Zykluszeit 2 ms.

| Szenario | Teile | Antwortzeit ms (Mittel / p95 / max) | Reaktion der SPS |
|---|---|---|---|
| Normalbetrieb | 200 | 35,8 / 42,6 / 48,7 | keine nötig |
| Kamera liefert jedes 10. Bild nicht | 50 | 33,5 / 45,3 / 46,3 | 5 Teile mit Fehlercode ausgeschleust (fail-safe) |
| Station fällt während eines Zyklus aus | 11 | 125,6 / 523,2 / 1000,5 | keine Antwort innerhalb der Überwachungszeit: Teil ausgeschleust, Linie gestoppt |
| Station fällt zwischen zwei Teilen aus | 5 | 35,7 / 40,0 / 40,2 | Lebenszeichen der Station fehlt: Linie gestoppt |

Normalbetrieb: davon Prüfung 29,9 ms, OPC-UA-Schreiben 2,5 ms; 1 Fehlschlupf und 0 Pseudoausschuss bei 200 Teilen.

### Kamerakalibrierung

Quelle: Selbsttest mit gerenderten Ansichten einer bekannten Kamera. Reprojektionsfehler 0,27 px (15 Ansichten). Positionsfehler auf der Arbeitsebene: 0,153 mm im Mittel (p95 0,493 mm) mit Linsenkorrektur, 0,346 mm ohne.

## 5. Bewertung und Empfehlung

**1. Saubere Bilder entscheiden nicht.** Auf sauberen Testbildern liegen alle Verfahren zwischen 92,8 % und 100,0 % Genauigkeit. Eine Vorführung unter Laborbedingungen zeigt nicht, welches Verfahren das richtige ist.

**2a. Stahloberfläche (6 Fehlerarten):** Bei veränderten Bedingungen liegt das robusteste Verfahren (vortrainierte CNN-Merkmale + linearer Klassifikator) im Mittel über alle 44 Bedingungen bei 79,6 %, das schwächste (klassische Merkmale + SVM) bei 49,4 %. Am schwierigsten: Unschärfe (Fokus), geringere Auflösung, Bewegungsunschärfe.

**2b. Gussteil (i. O. / n. i. O.):** Bei veränderten Bedingungen liegt das robusteste Verfahren (feinabgestimmtes CNN) im Mittel über alle 44 Bedingungen bei 89,9 %, das schwächste (klassische Merkmale + SVM) bei 67,0 %. Am schwierigsten: JPEG-Kompression, Sensorrauschen, Unschärfe (Fokus).

**3. Mit der erwarteten Variation trainieren.** Ein erneutes Training mit zufälliger Helligkeit, Unschärfe und Rauschen hebt die beste mittlere Genauigkeit unter veränderten Bedingungen von 79,6 % auf 94,4 % (Stahl) und von 89,9 % auf 96,8 % (Guss); der größte Gewinn beträgt +38 Prozentpunkte für klassische Merkmale + SVM (Stahl). Am besten bei beiden Aufgaben: feinabgestimmtes CNN, mit Variation trainiert (94,4 % Stahl, 96,8 % Guss).

**4. Wenige Trainingsbilder: das beste Verfahren hängt von der Aufgabe ab.** Mit nur 12 Bildern pro Klasse (Stahl) erreichen die vortrainierten Merkmale 97,2 %, das klassische Verfahren 95,2 % und das feinabgestimmte Netz 72,5 %. Bei den Gussteilen ist es umgekehrt: mit 18 Bildern pro Klasse erreicht das feinabgestimmte Netz 91,8 %, die vortrainierten Merkmale 80,3 %, das klassische Verfahren 76,3 %. Welches Verfahren mit den wenigsten Bildern auskommt, hängt von der Aufgabe ab und muss getestet werden.

**5. Keine Fehlerbilder? Anomalieerkennung.** Nur mit Gutteilen trainiert, trennt sie i. O. und n. i. O. mit AUROC 0,976 (303 Gutteilbilder); mit 50 Gutteilbildern 0,903, mit 5 nur 0,669. Mit ihrer automatischen Schwelle übersieht sie 7,3 % der Fehler und sortiert 7,1 % der Gutteile aus, deutlich schlechter als die überwachten Verfahren. Richtig für den Anlauf eines neuen Produkts, solange Fehlerbilder selten sind; sie reagiert aber auf jede Bildänderung, deshalb müssen Beleuchtung und Kamera stabil bleiben.

**6. Die Rechenzeit ist kein Engpass.** Jedes Verfahren braucht höchstens 40 ms pro Bild auf einer Laptop-CPU, weit unter der Taktzeit einer typischen Linie.

**7. Teilekennzeichnung:** auf sauberen Bildern lesen die Leser QR (OpenCV) 92 %, QR (ZBar) 100 %, QR (ZXing-C++) 97 %, Data Matrix (ZXing-C++) 93 %, Klartext (RapidOCR) 100 %. Am schwierigsten für alle Leser: Sensorrauschen, Bewegungsunschärfe, Unschärfe (Fokus).

**Empfehlung für eine neue Station:** zuerst Beleuchtung, Fokus und Kameramontage festlegen (veränderte Bedingungen kosten in den Versuchen weit mehr Genauigkeit als die Wahl zwischen den Verfahren auf sauberen Bildern), feinabgestimmtes CNN mit der erwarteten Variation trainieren, mit Anomalieerkennung starten, solange Fehlerbilder selten sind, und mit Versuchsbildern an der realen Station bestätigen.


## 6. Grenzen

- Die veränderten Bedingungen sind simuliert; sie ersetzen keine Versuche an der realen Station.
- Öffentliche Datensätze mit kontrollierter Aufnahme; reale Linien haben zusätzliche Störgrößen (Staub, Öl, Teilevarianten).
- Die Kennzeichnungen sind erzeugt (bekannter Inhalt), die Oberfläche ist echt.
- Ein Lauf mit festem Zufallswert; kleine Unterschiede (unter etwa 1 Prozentpunkt) sind nicht belastbar.
- SPS und Roboter sind simuliert (OPC UA auf demselben Rechner); im Netzwerk kommen Latenzen hinzu.
