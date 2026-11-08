**1. Saubere Bilder entscheiden nicht.** Auf sauberen Testbildern liegen alle Verfahren zwischen 92,8 % und 100,0 % Genauigkeit. Eine Vorführung unter Laborbedingungen zeigt nicht, welches Verfahren das richtige ist.

**2a. Stahloberfläche (6 Fehlerarten):** Bei veränderten Bedingungen liegt das robusteste Verfahren (vortrainierte CNN-Merkmale + linearer Klassifikator) im Mittel über alle 44 Bedingungen bei 79,6 %, das schwächste (klassische Merkmale + SVM) bei 49,4 %. Am schwierigsten: Unschärfe (Fokus), geringere Auflösung, Bewegungsunschärfe.

**2b. Gussteil (i. O. / n. i. O.):** Bei veränderten Bedingungen liegt das robusteste Verfahren (feinabgestimmtes CNN) im Mittel über alle 44 Bedingungen bei 89,9 %, das schwächste (klassische Merkmale + SVM) bei 67,0 %. Am schwierigsten: JPEG-Kompression, Sensorrauschen, Unschärfe (Fokus).

**3. Mit der erwarteten Variation trainieren.** Ein erneutes Training mit zufälliger Helligkeit, Unschärfe und Rauschen hebt die beste mittlere Genauigkeit unter veränderten Bedingungen von 79,6 % auf 94,4 % (Stahl) und von 89,9 % auf 96,8 % (Guss); der größte Gewinn beträgt +38 Prozentpunkte für klassische Merkmale + SVM (Stahl). Am besten bei beiden Aufgaben: feinabgestimmtes CNN, mit Variation trainiert (94,4 % Stahl, 96,8 % Guss).

**4. Wenige Trainingsbilder: das beste Verfahren hängt von der Aufgabe ab.** Mit nur 12 Bildern pro Klasse (Stahl) erreichen die vortrainierten Merkmale 97,2 %, das klassische Verfahren 95,2 % und das feinabgestimmte Netz 72,5 %. Bei den Gussteilen ist es umgekehrt: mit 18 Bildern pro Klasse erreicht das feinabgestimmte Netz 91,8 %, die vortrainierten Merkmale 80,3 %, das klassische Verfahren 76,3 %. Welches Verfahren mit den wenigsten Bildern auskommt, hängt von der Aufgabe ab und muss getestet werden.

**5. Keine Fehlerbilder? Anomalieerkennung.** Nur mit Gutteilen trainiert, trennt sie i. O. und n. i. O. mit AUROC 0,976 (303 Gutteilbilder); mit 50 Gutteilbildern 0,903, mit 5 nur 0,669. Mit ihrer automatischen Schwelle übersieht sie 7,3 % der Fehler und sortiert 7,1 % der Gutteile aus, deutlich schlechter als die überwachten Verfahren. Richtig für den Anlauf eines neuen Produkts, solange Fehlerbilder selten sind; sie reagiert aber auf jede Bildänderung, deshalb müssen Beleuchtung und Kamera stabil bleiben.

**6. Die Rechenzeit ist kein Engpass.** Jedes Verfahren braucht höchstens 40 ms pro Bild auf einer Laptop-CPU, weit unter der Taktzeit einer typischen Linie.

**7. Teilekennzeichnung:** auf sauberen Bildern lesen die Leser QR (OpenCV) 92 %, QR (ZBar) 100 %, QR (ZXing-C++) 97 %, Data Matrix (ZXing-C++) 93 %, Klartext (RapidOCR) 100 %. Am schwierigsten für alle Leser: Sensorrauschen, Bewegungsunschärfe, Unschärfe (Fokus).

**Empfehlung für eine neue Station:** zuerst Beleuchtung, Fokus und Kameramontage festlegen (veränderte Bedingungen kosten in den Versuchen weit mehr Genauigkeit als die Wahl zwischen den Verfahren auf sauberen Bildern), feinabgestimmtes CNN mit der erwarteten Variation trainieren, mit Anomalieerkennung starten, solange Fehlerbilder selten sind, und mit Versuchsbildern an der realen Station bestätigen.
